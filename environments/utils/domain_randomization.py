import numpy as np
import pybullet as p


class DomainRandomizationMixin:
    def __init__(
            self,
            *args,
            dr_on_reset: bool = True,
            dr_mass_range: tuple = (0.95, 1.05),
            dr_kf_range: tuple = (0.95, 1.05),
            dr_km_range: tuple = (0.95, 1.05),
            dr_arm_length_range: tuple = (0.95, 1.05),
            dr_scale_inertia: bool = True,
            dr_seed: int | None = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "M") and hasattr(self, "MASS"):
            self.M = float(self.MASS)

        self._dr_nominal_mass = float(self.M)
        self._dr_nominal_kf = float(self.KF)
        self._dr_nominal_km = float(self.KM)
        self._dr_nominal_arm_length = float(getattr(self, "L", 0.0))

        self._dr_on_reset = bool(dr_on_reset)
        self._dr_mass_range = tuple(dr_mass_range)
        self._dr_kf_range = tuple(dr_kf_range)
        self._dr_km_range = tuple(dr_km_range)
        self._dr_arm_length_range = tuple(dr_arm_length_range)
        self._dr_scale_inertia = bool(dr_scale_inertia)
        self._dr_random_generator = np.random.default_rng(dr_seed)

        self._recompute_weight_and_hover()
        self._dr_last = None

    def _set_mass_in_bullet(self, mass: float, scale_inertia: float | None = None):
        for uid in getattr(self, "DRONE_IDS", []):
            p.changeDynamics(uid, -1, mass=float(mass), physicsClientId=self.CLIENT)
            if scale_inertia is not None:
                dynamics_info = p.getDynamicsInfo(uid, -1, physicsClientId=self.CLIENT)[2]
                p.changeDynamics(
                    uid, -1,
                    localInertiaDiagonal = (np.asarray(dynamics_info) * float(scale_inertia)).tolist(),
                    physicsClientID = self.CLIENT
                )

    def _refresh_arm_dependent_terms(self, old_L: float | None = None):
        if not hasattr(self, "L") or (old_L is None) or old_L == 0.0:
            old_L = float(self._dr_nominal_arm_length) if getattr(self, "_dr_nominal_arm_length", 0.0) > 0.0 else None

        if old_L is not None or old_L == 0.0:
            ratio = None
        else:
            ratio = float(self.L) / float(old_L) if float(old_L) != 0.0 else None

    def set_dr_params(
            self,
            *,
            mass: float | None = None,
            kf: float | None = None,
            km: float | None = None,
            arm_length: float | None = None,
            scale_inertia: float | None = None,
            update_log: bool = True
    ):
        if mass is not None:
            self._set_mass_in_bullet(mass, scale_inertia=scale_inertia)
            self.M = float(mass)
        if kf is not None:
            self.KF = float(kf)
        if km is not None:
            self.KM = float(km)
        if arm_length is not None and hasattr(self, "L"):
            self.L = float(arm_length)

        self._dr_after_change()
        if update_log:
            self._dr_last = {
                "mass": float(self.M),
                "kf": float(self.KF),
                "km": float(self.KM),
                "arm_length": float(self.L) if hasattr(self, "L") else self._dr_nominal_arm_length,
                "hover_rpm": float(self.HOVER_RPM)
            }

    def get_last_dr_params(self):
        return dict(self._dr_last) if self._dr_last is not None else None

    def _recompute_weight_and_hover(self):
        if hasattr(self, "G"):
            self.GRAVITY = float(self.G) * float(self.M)

        self.HOVER_RPM = float(np.sqrt(float(self.GRAVITY) / float((4.0 * self.KF))))

    def _dr_sample_scales(self):
        sample_mass = self._dr_random_generator.uniform(*self._dr_mass_range)
        sample_kf = self._dr_random_generator.uniform(*self._dr_kf_range)
        sample_km = self._dr_random_generator.uniform(*self._dr_km_range)
        sample_arm_length = self._dr_random_generator.uniform(*self._dr_arm_length_range)
        return sample_mass, sample_kf, sample_km, sample_arm_length

    def _dr_after_change(self):
        self._recompute_weight_and_hover()
        if hasattr(self, "RPM2FORCE"):
            self.RPM2FORCE = self.KF

        if hasattr(self, "_rpm2force"):
            self._rpm2force = self.KF

    def _apply_domain_randomization(self):
        sm, skf, skm, sal = self._dr_sample_scales()
        scaled_mass = self._dr_nominal_mass * sm
        scaled_kf = self._dr_nominal_kf * skf
        scaled_km = self._dr_nominal_km * skm
        scaled_arm_length = (self._dr_nominal_arm_length * sal) if (hasattr(self, "L") and self._dr_nominal_arm_length
                                                                    > 0.0) else None

        self.set_dr_params(
            mass=scaled_mass,
            kf=scaled_kf,
            km=scaled_km,
            arm_length=scaled_arm_length,
            scale_inertia=(sm if self._dr_scale_inertia else None),
            update_log=True
        )

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        if self._dr_on_reset:
            self._apply_domain_randomization()
            info = dict(info or {})
            info["dr_params"] = dict(self._dr_last)

        return obs, info
