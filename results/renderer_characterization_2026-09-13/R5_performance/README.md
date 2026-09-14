# RC-R5 — render-path cost (descriptive)

Verdict: **DESCRIPTIVE_NO_VERDICT**. RC-R5 has no pass/fail threshold.

54 of 54 cells were measured; the rest hit the preregistered resource stop rule.

| configuration | resolution | scenes | headline median ms | staged median ms | instrumentation ms |
|---|---|---|---|---|---|
| `analytic_sphere_geometry` | 160x90 | 1 | 1.142 | 1.140 | -0.002 |
| `analytic_sphere_geometry` | 160x90 | 8 | 9.228 | 9.174 | -0.054 |
| `analytic_sphere_geometry` | 160x90 | 32 | 37.500 | 37.869 | 0.369 |
| `analytic_sphere_geometry` | 320x240 | 1 | 5.771 | 5.802 | 0.031 |
| `analytic_sphere_geometry` | 320x240 | 8 | 50.651 | 50.665 | 0.014 |
| `analytic_sphere_geometry` | 320x240 | 32 | 217.986 | 219.159 | 1.172 |
| `analytic_sphere_geometry` | 640x480 | 1 | 26.064 | 26.052 | -0.012 |
| `analytic_sphere_geometry` | 640x480 | 8 | 219.814 | 221.183 | 1.368 |
| `analytic_sphere_geometry` | 640x480 | 32 | 933.114 | 935.524 | 2.410 |
| `box_geometry` | 160x90 | 1 | 0.584 | 0.518 | -0.066 |
| `box_geometry` | 160x90 | 8 | 0.682 | 0.621 | -0.061 |
| `box_geometry` | 160x90 | 32 | 0.903 | 0.867 | -0.036 |
| `box_geometry` | 320x240 | 1 | 0.888 | 0.824 | -0.064 |
| `box_geometry` | 320x240 | 8 | 1.270 | 1.257 | -0.013 |
| `box_geometry` | 320x240 | 32 | 3.161 | 3.180 | 0.019 |
| `box_geometry` | 640x480 | 1 | 2.013 | 1.965 | -0.048 |
| `box_geometry` | 640x480 | 8 | 4.232 | 4.169 | -0.062 |
| `box_geometry` | 640x480 | 32 | 12.739 | 12.654 | -0.085 |
| `box_flat` | 160x90 | 1 | 0.588 | 0.515 | -0.073 |
| `box_flat` | 160x90 | 8 | 0.682 | 0.617 | -0.065 |
| `box_flat` | 160x90 | 32 | 0.910 | 0.874 | -0.036 |
| `box_flat` | 320x240 | 1 | 0.889 | 0.816 | -0.073 |
| `box_flat` | 320x240 | 8 | 1.262 | 1.246 | -0.016 |
| `box_flat` | 320x240 | 32 | 3.164 | 3.180 | 0.016 |
| `box_flat` | 640x480 | 1 | 1.991 | 1.928 | -0.063 |
| `box_flat` | 640x480 | 8 | 4.314 | 4.221 | -0.092 |
| `box_flat` | 640x480 | 32 | 12.795 | 12.680 | -0.115 |
| `area_matched_box_lambertian` | 160x90 | 1 | 0.588 | 0.520 | -0.067 |
| `area_matched_box_lambertian` | 160x90 | 8 | 0.686 | 0.625 | -0.061 |
| `area_matched_box_lambertian` | 160x90 | 32 | 0.884 | 0.854 | -0.029 |
| `area_matched_box_lambertian` | 320x240 | 1 | 0.888 | 0.818 | -0.070 |
| `area_matched_box_lambertian` | 320x240 | 8 | 1.273 | 1.243 | -0.031 |
| `area_matched_box_lambertian` | 320x240 | 32 | 3.136 | 3.151 | 0.015 |
| `area_matched_box_lambertian` | 640x480 | 1 | 2.038 | 1.950 | -0.088 |
| `area_matched_box_lambertian` | 640x480 | 8 | 4.186 | 4.158 | -0.027 |
| `area_matched_box_lambertian` | 640x480 | 32 | 12.608 | 12.710 | 0.102 |
| `quadrotor_geometry` | 160x90 | 1 | 0.789 | 0.732 | -0.057 |
| `quadrotor_geometry` | 160x90 | 8 | 1.034 | 0.969 | -0.065 |
| `quadrotor_geometry` | 160x90 | 32 | 1.392 | 1.359 | -0.032 |
| `quadrotor_geometry` | 320x240 | 1 | 1.129 | 1.069 | -0.061 |
| `quadrotor_geometry` | 320x240 | 8 | 1.681 | 1.657 | -0.024 |
| `quadrotor_geometry` | 320x240 | 32 | 3.935 | 3.982 | 0.046 |
| `quadrotor_geometry` | 640x480 | 1 | 2.321 | 2.226 | -0.095 |
| `quadrotor_geometry` | 640x480 | 8 | 4.854 | 4.902 | 0.049 |
| `quadrotor_geometry` | 640x480 | 32 | 14.321 | 14.357 | 0.036 |
| `quadrotor_lambertian` | 160x90 | 1 | 0.802 | 0.733 | -0.069 |
| `quadrotor_lambertian` | 160x90 | 8 | 1.037 | 0.975 | -0.062 |
| `quadrotor_lambertian` | 160x90 | 32 | 1.392 | 1.356 | -0.036 |
| `quadrotor_lambertian` | 320x240 | 1 | 1.132 | 1.066 | -0.066 |
| `quadrotor_lambertian` | 320x240 | 8 | 1.677 | 1.667 | -0.010 |
| `quadrotor_lambertian` | 320x240 | 32 | 4.108 | 3.935 | -0.173 |
| `quadrotor_lambertian` | 640x480 | 1 | 2.336 | 2.238 | -0.098 |
| `quadrotor_lambertian` | 640x480 | 8 | 4.825 | 4.752 | -0.073 |
| `quadrotor_lambertian` | 640x480 | 32 | 14.370 | 14.347 | -0.022 |

Not comparable with D6 (1.5338x ray-transform ratio, INCONCLUSIVE) or D7 (+0.411 ms integrated shadow cost, GO): different denominators and different code paths.

Memory is reported as five independent fields per cell. A field that could not be
read says `UNAVAILABLE` and carries its reason; none is reported as zero.
