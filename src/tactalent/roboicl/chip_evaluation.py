"""Disclosed evaluation-only chip rules; velocity-triggered fracture is diagnostic."""

from dataclasses import replace

PROTOCOL = "roboicl_chip_stable_placement_no_impact_v5"
RECOVERABLE = frozenset(
    {
        "early_release",
        "slip",
        "unstable_grasp",
        "off_plate",
        "gripper_orientation_violation",
        "post_success_state_changed",
    }
)


def evaluation_rules(task):
    return dict(
        protocol=PROTOCOL,
        upstream_protocol="chip_physical_lifecycle_v3",
        matches_upstream_protocol=False,
        support_before_release_required=False,
        plate_overpress_enforced=False,
        impact_fracture_enforced=False,
        release_required=False,
        withdrawal_required=False,
        gripper_orientation_enforced=False,
        in_target_slip_preserves_verified_grasp_lift=True,
        recoverable_events=sorted(RECOVERABLE),
        max_control_steps=int(task.cfg.step_lim),
        max_policy_seconds=float(task.cfg.final_policy_timeout_seconds),
        budgets_unchanged=False,
        native_max_control_steps=3600,
        budget_basis="user_reviewed_600_commands; 405-command training demonstration",
        modification_record="docs/roboicl_chip_evaluation_protocol.md",
    )


class DiagnosticImpactTracker:
    """Keep native landing detection while suppressing its speed-only fracture."""

    def __init__(self, native_tracker, task):
        self.native_tracker = native_tracker
        self.task = task

    def __getattr__(self, name):
        return getattr(self.native_tracker, name)

    def advance(self, incoming_downward_speed, upward_contact, gap_m):
        would_fracture = self.native_tracker.advance(incoming_downward_speed, upward_contact, gap_m)
        if would_fracture:
            self.task.metadata.setdefault("impact_threshold_events", []).append(
                dict(
                    step=int(self.task.step_count),
                    reason="impact",
                    incoming_speed_m_s=float(incoming_downward_speed),
                    critical_speed_m_s=float(self.native_tracker.critical_speed_m_s),
                    gap_m=float(gap_m),
                    upward_contact=bool(upward_contact),
                    native_threshold_exceeded=True,
                    fracture_applied=False,
                    protocol=PROTOCOL,
                )
            )
        # Returning False lets the native integrator update its velocity history
        # and evaluate strain damage in this same step; don't undo a fracture
        # after the native code has already swapped the chip for fragments.
        return False


def configure_evaluation_budget(cfg, spec, mode):
    """Apply the disclosed command/time budget only to evaluation instances."""
    if mode != "eval":
        return
    maximum = int(spec["evaluation_action_limit"])
    if maximum <= 0:
        raise ValueError("Chip evaluation command budget must be positive")
    cfg.step_lim = maximum
    cfg.final_policy_timeout_seconds = (
        maximum * float(cfg.sim.dt) * cfg.decimation * cfg.policy_action_repeat
    )


def recovery_scorer(native_scorer, observer):
    """Retain native physical evidence, changing only the reviewed event policy."""
    if native_scorer.last_step != -1:
        raise RuntimeError("Install chip evaluation rules before the first policy step")

    class RecoverableChipLifecycle(type(native_scorer)):
        acceptance_protocol = PROTOCOL

        def __init__(self, limits):
            super().__init__(limits)
            self.nonterminal_events = []
            self.stable_placement_ticks = 0

        def advance(self, sample):
            # Bypass the gate itself: fail() alone would still return early
            # and discard valid placement evidence. Keep real sample values.
            previous_step = self.last_step
            previous_stage = self.stage
            limits = self.limits
            self.limits = replace(limits, overtravel_m=float("inf"), pose_error_rad=float("inf"))
            try:
                result = super().advance(sample)
            finally:
                self.limits = limits
            if (
                sample.step != previous_step
                and not self.failure
                and sample.pose_error_rad > limits.pose_error_rad
            ):
                self.nonterminal_events.append(
                    dict(
                        step=sample.step,
                        reason="gripper_orientation_violation",
                        previous_stage=previous_stage,
                        recovery="diagnostic_only",
                    )
                )
            if sample.step != previous_step and not self.success:
                qualifies = (
                    not self.failure
                    and self.grasp_verified
                    and self.lift_verified
                    and self._stable_supported(sample)
                )
                self.stable_placement_ticks = self.stable_placement_ticks + 1 if qualifies else 0
                if self.stable_placement_ticks >= self.limits.support_ticks:
                    self.outcome = "success"
                    self._transition("terminal")
            return result

        def fail(self, reason, outcome="failure"):
            fractured = bool(getattr(observer.task, "fractured", False)) or bool(
                getattr(self.last_sample, "fractured", False)
            )
            if fractured or self.failure:
                return super().fail(reason, outcome)
            if (
                reason == "post_success_state_changed"
                and self.success
                and self._stable_supported(self.last_sample)
            ):
                # Native completion also requires release/withdrawal. V3 keeps
                # the real sample but no longer revokes a stable placement for
                # either of those gripper-only conditions.
                return
            if reason in ("invalid_physics", "error"):
                super().fail(reason, "invalid")
                raise RuntimeError("OpenViTac chip physics error: " + reason)
            # Preserve policy_timeout and fail closed on unreviewed reasons.
            if reason not in RECOVERABLE:
                return super().fail(reason, outcome)
            self.nonterminal_events.append(
                dict(step=self.last_step, reason=reason, previous_stage=self.stage)
            )
            if reason == "gripper_orientation_violation":
                # Orientation is diagnostic: it must not erase physical grasp,
                # lift, support dwell, or a previously confirmed placement.
                self.nonterminal_events[-1]["recovery"] = "diagnostic_only"
                return
            self.outcome = "pending"
            if (
                reason in ("early_release", "slip")
                and self.grasp_verified
                and self.lift_verified
                and self.last_sample.in_target
            ):
                # A lost pad contact over the target is recoverable just like
                # early opening: it must not erase an already verified
                # grasp/lift. Stable intact tray support is still required.
                # Outside-target slips keep the new-attempt reset below.
                self.released_ticks = self.withdrawn_ticks = 0
                self.release_supported = False
                self.nonterminal_events[-1]["recovery"] = "await_stable_placement"
                self._transition("release")
                return
            self.stable_placement_ticks = 0
            self.grip_ticks = self.lift_ticks = self.support_ticks = 0
            self.released_ticks = self.withdrawn_ticks = 0
            self.grasp_verified = self.lift_verified = self.release_supported = False
            observer.reference = observer.grip_qpos = observer.support_ee_z = None
            self._transition("approach")

        def finish(self, reason="incomplete"):
            if self.outcome == "pending":
                outcome = (
                    "invalid" if reason in ("error", "timeout", "invalid_physics") else "failure"
                )
                super().fail(reason, outcome)

        def snapshot(self):
            return dict(
                super().snapshot(),
                acceptance_protocol=PROTOCOL,
                plate_overpress_enforced=False,
                impact_fracture_enforced=False,
                release_required=False,
                withdrawal_required=False,
                gripper_orientation_enforced=False,
                in_target_slip_preserves_verified_grasp_lift=True,
                stable_placement_ticks=self.stable_placement_ticks,
                nonterminal_events=[dict(row) for row in self.nonterminal_events],
            )

    return RecoverableChipLifecycle(native_scorer.limits)


def evaluation_task_class(native, native_monitor):
    """Use only for eval instances; expert collection keeps its native scorer."""

    class EvaluationChipTask(native):
        def _record_chip_protocol(self):
            observer = getattr(self, "_action_monitor", None)
            if (
                observer is not None
                and getattr(observer.scorer, "acceptance_protocol", None) == PROTOCOL
            ):
                self.metadata["evaluation_rules"] = evaluation_rules(self)
                self.metadata["success_contract"] = PROTOCOL
                self.metadata["success_diagnostics"] = observer.scorer.snapshot()
                self.metadata["physical_result"] = observer.scorer.outcome
                self.metadata["terminal_reason"] = observer.scorer.snapshot()["terminal_reason"]

        def _install_chip_protocol(self):
            observer = getattr(self, "_action_monitor", None)
            if observer is None:
                observer = self._action_monitor = native_monitor(self)
            if getattr(observer.scorer, "acceptance_protocol", None) != PROTOCOL:
                observer.scorer = recovery_scorer(observer.scorer, observer)
            self._record_chip_protocol()

        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            self._install_chip_protocol()
            return result

        def take_action(self, *args, **kwargs):
            # env_step uses this native path, retaining command registration,
            # force-task repetition, accounting and finalization.
            self._install_chip_protocol()
            try:
                return super().take_action(*args, **kwargs)
            finally:
                self._record_chip_protocol()

        def clean_cache(self, mean_steps=0.0, result=None):
            # Native eval cleanup saves metadata but does not call the training
            # save_to_hdf5 finalizer. Close incomplete evaluation evidence here.
            observer = getattr(self, "_action_monitor", None)
            if (
                result is not None
                and observer is not None
                and getattr(observer.scorer, "acceptance_protocol", None) == PROTOCOL
            ):
                observer.scorer.finish("error" if result == "error" else "incomplete")
                self._record_chip_protocol()
            return super().clean_cache(mean_steps=mean_steps, result=result)

        def _advance_chip_physics(self, *args, **kwargs):
            # Native reset creates a fresh tracker. Wrap that instance, without
            # changing the native class, constants, training or other tasks.
            tracker = getattr(self, "_chip_impact_tracker", None)
            if tracker is not None and not isinstance(tracker, DiagnosticImpactTracker):
                self._chip_impact_tracker = DiagnosticImpactTracker(tracker, self)
            return super()._advance_chip_physics(*args, **kwargs)

        def _step(self, *args, **kwargs):
            try:
                return super()._step(*args, **kwargs)
            finally:
                # The native monitor writes a literal protocol name per step.
                self._record_chip_protocol()

    EvaluationChipTask.__name__ = native.__name__
    EvaluationChipTask.__module__ = native.__module__
    return EvaluationChipTask
