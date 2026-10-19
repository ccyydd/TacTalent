"""OpenViTac chip reset: align unloaded FEM pads before native world rebuild.

The robot is reset by PhysX while UIPC keeps the authored gel positions.
Do not ask the finite-element solver to drag a pad through the scene to catch
up with a teleported gripper. Use the native attachment offsets/rest mesh at
reset only; subsequent contact dynamics and all acceptance rules stay native.
"""

import numpy as np


def attachment_residuals(task):
    rows = {}
    for name, sensor in task._tactile_manager.tactiles.items():
        attachment = sensor.attachment
        points = np.asarray(sensor.gelpad.geo_slot_list[0].geometry().positions().view()).reshape(
            -1, 3
        )
        targets = attachment._compute_aim_positions()
        ids = attachment.attachment_points_idx
        if not len(ids) or not np.isfinite(points).all() or not np.isfinite(targets).all():
            raise RuntimeError("invalid_physics: chip gel attachment data is invalid")
        errors = np.linalg.norm(points[ids] - targets, axis=1)
        rows[name] = {"max_error_m": float(errors.max()), "mean_error_m": float(errors.mean())}
    return rows


def align_unloaded_pads(task):
    from uipc import view

    rows = {}
    for name, sensor in task._tactile_manager.tactiles.items():
        geometry = sensor.gelpad.geo_slot_list[0].geometry()
        target = sensor._get_gelpad_points_for_current_attachment()
        if target is None:
            raise RuntimeError("invalid_physics: chip gel rest mesh is unavailable")
        target = target.detach().cpu().numpy().astype(np.float64)
        prior = np.asarray(geometry.positions().view()).reshape(-1, 3).copy()
        if target.shape != prior.shape or not np.isfinite(target).all():
            raise RuntimeError("invalid_physics: chip gel rest mesh is invalid")
        view(geometry.positions())[:] = target.reshape(geometry.positions().view().shape)
        aim = geometry.vertices().find("aim_position")
        if aim is None:
            raise RuntimeError("invalid_physics: chip gel constraint targets are unavailable")
        view(aim)[:] = target.reshape(aim.view().shape)
        ids = sensor.attachment.attachment_points_idx
        sensor.attachment.aim_positions = target[ids].copy()
        velocity = geometry.vertices().find("velocity")
        if velocity is not None:
            view(velocity)[:] = 0
        rows[name] = {"max_reset_shift_m": float(np.linalg.norm(prior - target, axis=1).max())}
    return rows


def task_class(native):
    class AlignedChipTask(native):
        def _reset_actors(self):
            # Native _reset_actors rebuilds UIPC with zero velocities after
            # staging actors. Include the correctly positioned gels in that
            # same rebuild, before settling and tactile reference capture.
            alignment = align_unloaded_pads(self)
            result = super()._reset_actors()
            self.metadata["roboicl_chip_gel_reset_alignment"] = alignment
            return result

        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            residuals = attachment_residuals(self)
            self.metadata["roboicl_chip_gel_attachment_residuals"] = residuals
            # Startup validity only: no contact should exist here. This is not
            # a policy failure rule and is never applied during manipulation.
            if any(row["max_error_m"] > 0.001 for row in residuals.values()):
                raise RuntimeError("invalid_physics: chip gel pads detached during initialization")
            return result

    AlignedChipTask.__name__ = native.__name__
    AlignedChipTask.__module__ = native.__module__
    return AlignedChipTask
