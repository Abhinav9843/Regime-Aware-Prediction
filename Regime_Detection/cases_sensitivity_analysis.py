CONFIGS = [
    # ---------- baseline ----------
    dict(name="BASE",
         alpha0_a_pri=2.0, alpha0_b_pri=1.0,
         gamma0_a_pri=3.0, gamma0_b_pri=1.0,
         v0_range=(0.84, 0.86),   # E[kappa] ~ 0.85
         v1_range=(0.35, 0.39),   # concentration ~ 1/v1^p ~ 20 (p=3)
         p=3),

    # ---------- alpha sensitivity ----------
    dict(name="ALPHA_LOW",
         alpha0_a_pri=1.0, alpha0_b_pri=1.0,
         gamma0_a_pri=3.0, gamma0_b_pri=1.0,
         v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),

    dict(name="ALPHA_HIGH",
         alpha0_a_pri=4.0, alpha0_b_pri=1.0,
         gamma0_a_pri=3.0, gamma0_b_pri=1.0,
         v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),

    # ---------- gamma sensitivity ----------
    dict(name="GAMMA_LOW",
         alpha0_a_pri=2.0, alpha0_b_pri=1.0,
         gamma0_a_pri=1.5, gamma0_b_pri=1.0,
         v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),

    dict(name="GAMMA_HIGH",
         alpha0_a_pri=2.0, alpha0_b_pri=1.0,
         gamma0_a_pri=6.0, gamma0_b_pri=1.0,
         v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),

    # ---------- stickiness (rho1,rho2) sensitivity ----------
    # low stickiness: E[kappa] ~ 0.71, concentration ~ 10
    dict(name="STICK_LOW",
         alpha0_a_pri=2.0, alpha0_b_pri=1.0,
         gamma0_a_pri=3.0, gamma0_b_pri=1.0,
         v0_range=(0.70, 0.72),v1_range=(0.44, 0.48), p=3),

    # high stickiness: E[kappa] ~ 0.95, concentration ~ 40
    dict(name="STICK_HIGH",
         alpha0_a_pri=2.0, alpha0_b_pri=1.0,
         gamma0_a_pri=3.0, gamma0_b_pri=1.0,
         v0_range=(0.94, 0.96),v1_range=(0.28, 0.31), p=3),

    # ---------- corner combos ----------
    dict(name="LOW_LOW",
         alpha0_a_pri=1.0, alpha0_b_pri=1.0,
         gamma0_a_pri=1.5, gamma0_b_pri=1.0,
         v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),

    dict(name="HIGH_HIGH",
         alpha0_a_pri=4.0, alpha0_b_pri=1.0,
         gamma0_a_pri=6.0, gamma0_b_pri=1.0,
         v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),

    dict(name="ALPHA_HIGH_STICK_HIGH",
         alpha0_a_pri=4.0, alpha0_b_pri=1.0,
         gamma0_a_pri=3.0, gamma0_b_pri=1.0,
         v0_range=(0.94, 0.96), v1_range=(0.28, 0.31), p=3),

    dict(name="GAMMA_HIGH_STICK_LOW",
         alpha0_a_pri=2.0, alpha0_b_pri=1.0,
         gamma0_a_pri=6.0, gamma0_b_pri=1.0,
         v0_range=(0.70, 0.72), v1_range=(0.44, 0.48), p=3),
]
