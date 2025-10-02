# With all features

[1] "--- Training on fold 3 ---"
[1] "Creating spatial weight matrix for training df..."
[1] "Fitting model..."
Error: cannot allocate vector of size 67.4 Mb
In addition: There were 40 warnings (use warnings() to see them)

> Warning messages:
> 1: In lagsarlm(very_good_health ~ ., data = df, listw = w) :
> Aliased variables found: good_health fair_health bad_health very_bad_health f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33 f34 f35 f36 f37 f38 f39 f40 f41 f42 f43 f44 f45 f46 f47 f48 f49 f50 f51 f52 f53 f54 f55 f56 f57 f58 f59 f60 f61 f62 f63 f64 f65 f66 f67 f68 f69 f70 f71 f72 f73 f74 f75 f76 f77 f78 f79 f80 f81 f82 f83 f84 f85 f86 f87 f88 f89 f90 m0 m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32 m33 m34 m35 m36 m37 m38 m39 m40 m41 m42 m43 m44 m45 m46 m47 m48 m49 m50 m51 m52 m53 m54 m55 m56 m57 m58 m59 m60 m61 m62 m63 m64 m65 m66 m67 m68 m69 m70 m71 m72 m73 m74 m75 m76 m77 m78 m79 m80 m81 m82 m83 m84 m85 m86 m87 m88 m89 m90 white_british white_irish white_gypsy.irish_traveller white_roma white_other mixed_white_and_asian mixed_white_and_black_african mixed_white_and_black_caribbean mixed_other asian_bangladeshi asian_chinese asian_india [... truncated]
> 2: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 3: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 4: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 5: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 6: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 7: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 8: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 9: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 10: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 11: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 12: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 13: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 14: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 15: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 16: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 17: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 18: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 19: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 20: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 21: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 22: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 23: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 24: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 25: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 26: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 27: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 28: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 29: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 30: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 31: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 32: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 33: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 34: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 35: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 36: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 37: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 38: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 39: In optimize(sar.lag.mixed.f, interval = interval, maximum = TRUE, ... :
> NA/NaN replaced by maximum positive value
> 40: In lagsarlm(very_good_health ~ ., data = df, listw = w) :
> inversion of asymptotic covariance matrix failed for tol.solve = 2.22044604925031e-16
> system is exactly singular - using numerical Hessian.

---

# With LSOA, health, fold_ids and geometry removed

[1] "--- Training on fold 3 ---"
[1] "Creating spatial weight matrix for training df..."
[1] "Fitting model..."

Spatial lag model
Warning in lagsarlm(very_good_health ~ ., data = df, listw = w, method = "eigen", :
Aliased variables found: m90 any_other
Jacobian calculated using neighbourhood matrix eigenvalues
Computing eigenvalues ...

rho: -0.4728035 function value: 6257.521
rho: 0.08975737 function value: 7252.612
rho: 0.4374391 function value: 7378.274
rho: 0.3804732 function value: 7398.013
rho: 0.3378087 function value: 7401.673
rho: 0.3427516 function value: 7401.728
rho: 0.3424438 function value: 7401.728
rho: 0.3424632 function value: 7401.728
rho: 0.3424633 function value: 7401.728
rho: 0.3424633 function value: 7401.728
rho: 0.3424633 function value: 7401.728
rho: 0.3424633 function value: 7401.728
rho: 0.3424633 function value: 7401.728
rho: 0.3424633 function value: 7401.728
rho: 0.3424633 function value: 7401.728
Warning in lagsarlm(very_good_health ~ ., data = df, listw = w, method = "eigen", :
inversion of asymptotic covariance matrix failed for tol.solve = 2.22044604925031e-16
reciprocal condition number = 6.41503e-22 - using numerical Hessian.

Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728

...

Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728
Hessian: rho: 0.3424633 function value: 7401.728

# With feature engineering

> # Build model
>
> model <- build_model(train_df)
> [1] "Creating spatial weight matrix for training df..."
> [1] "Fitting model..."

Spatial lag model
Warning in lagsarlm(very_good_health ~ ., data = df, listw = w, quiet = FALSE, :
Aliased variables found: prevalent_white_otherTRUE
Jacobian calculated using neighbourhood matrix eigenvalues
Computing eigenvalues ...

rho: -0.4728035 function value: 5657.393
rho: 0.08975737 function value: 7025.825
rho: 0.4374391 function value: 7593.576
rho: 0.08975737 function value: 7025.825
rho: 0.4374391 function value: 7593.576
rho: 0.6523183 function value: 7622.78
rho: 0.570414 function value: 7651.455
rho: 0.5634709 function value: 7651.48
rho: 0.5664794 function value: 7651.513
rho: 0.5664595 function value: 7651.513
rho: 0.5664625 function value: 7651.513
rho: 0.5664625 function value: 7651.513
rho: 0.5664625 function value: 7651.513
rho: 0.566469 function value: 7651.513
rho: 0.566465 function value: 7651.513
rho: 0.5664634 function value: 7651.513
rho: 0.5664629 function value: 7651.513
rho: 0.5664626 function value: 7651.513
rho: 0.5664625 function value: 7651.513
rho: 0.5664625 function value: 7651.513
rho: 0.5664625 function value: 7651.513
Warning in lagsarlm(very_good_health ~ ., data = df, listw = w, quiet = FALSE, :
inversion of asymptotic covariance matrix failed for tol.solve = 2.22044604925031e-16
reciprocal condition number = 1.3291e-19 - using numerical Hessian.
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664659 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513

...

Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Error in solve.default(-(mat), tol.solve = tol.solve) :
system is computationally singular: reciprocal condition number = 1.12396e-20
Timing stopped at: 146.2 0.65 147.9

# With prevalent_white_other removed

> model <- build_model(train_df)
> [1] "Creating spatial weight matrix for training df..."
> [1] "Fitting model..."

Spatial lag model
Jacobian calculated using neighbourhood matrix eigenvalues
Computing eigenvalues ...

rho: -0.4728035 function value: 5657.393
rho: 0.08975737 function value: 7025.825
rho: 0.4374391 function value: 7593.576
rho: 0.6523183 function value: 7622.78
rho: 0.570414 function value: 7651.455
rho: 0.5634709 function value: 7651.48
rho: 0.5664794 function value: 7651.513
rho: 0.5664595 function value: 7651.513
rho: 0.5664625 function value: 7651.513
rho: 0.5664625 function value: 7651.513
rho: 0.5664625 function value: 7651.513
rho: 0.566469 function value: 7651.513
rho: 0.566465 function value: 7651.513
rho: 0.5664634 function value: 7651.513
rho: 0.5664629 function value: 7651.513
rho: 0.5664626 function value: 7651.513
rho: 0.5664625 function value: 7651.513
rho: 0.5664625 function value: 7651.513
rho: 0.5664625 function value: 7651.513
Warning in lagsarlm(very_good_health ~ ., data = df, listw = w, quiet = FALSE, :
inversion of asymptotic covariance matrix failed for tol.solve = 2.22044604925031e-16
reciprocal condition number = 1.3291e-19 - using numerical Hessian.
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664659 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513

...

Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Hessian: rho: 0.5664625 function value: 7651.513
Error in solve.default(-(mat), tol.solve = tol.solve) :
system is computationally singular: reciprocal condition number = 1.12396e-20
Timing stopped at: 137.4 0.44 138.5

# With method = distance and small distance values

neighbour object has subgraphs

# With method = distance and distance ~2000

neighbour object has 5 sub-graphs
