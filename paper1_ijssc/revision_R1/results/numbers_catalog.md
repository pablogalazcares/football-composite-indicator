# Catálogo de cifras (SPEC §6.1)

Las cifras sobre la versión enviada están en `evidence_submitted_version.md`.

| Comentario | Qué | Clave → macro → valor |
|---|---|---|
| M1 | ρk enviado reproducido; implícito de M2 | `submitted.rhoK1` → `\RSubmittedRhoKOne` → 0.94<br>`submitted.rhoK2` → `\RSubmittedRhoKTwo` → 0.72<br>`implied.M2.k1` → `\RImpliedMTwoKOne` → 0.993<br>`implied.M2.k2` → `\RImpliedMTwoKTwo` → 0.951 |
| M2 | Persistencia observada a k = 1 y k = 2 (ρ_W) | `persist.BC.rhoW.est` → `\RPersistBCRhoWEst` → 0.51<br>`persist.AC.rhoW.est` → `\RPersistACRhoWEst` → 0.29 |
| M2/M5 | Precisión: score actual frente al modelo con edad (Δρ_W) | `acc.BC.M2.dRhoW.est` → `\RAccBCMTwoDRhoWEst` → $-$0.004<br>`acc.AC.M2.dRhoW.est` → `\RAccACMTwoDRhoWEst` → $-$0.033 |
| M3 | Efectos de edad de M2 con IC | `model.ageC` → `\RModelAgeC` → $-$0.029<br>`model.ageC.lo` → `\RModelAgeCLo` → $-$0.467<br>`model.ageC.hi` → `\RModelAgeCHi` → 0.324 |
| M7 | Minutos inflados en lo enviado; IPW | `submitted.minutesInflationMedian` → `\RSubmittedMinutesInflationMedian` → 1.23<br>`submitted.pairsTrueMinutes` → `\RSubmittedPairsTrueMinutes` → 160<br>`ipw.BC.rhoW` → `\RIpwBCRhoW` → 0.50<br>`ipw.AC.rhoW` → `\RIpwACRhoW` → 0.33 |
| M9 | Validez externa | `ext.minutes.BC.rhoW` → `\RExtMinutesBCRhoW` → 0.09<br>`ext.auc.BC.est` → `\RExtAucBCEst` → 0.51<br>`ext.mv.logmvB.est` → `\RExtMvLogmvBEst` → 0.09 |
| m2 | ρ por posición | `persist.BC.rhoDEF.est` → `\RPersistBCRhoDEFEst` → 0.44<br>`persist.BC.rhoMID.est` → `\RPersistBCRhoMIDEst` → 0.55<br>`persist.BC.rhoFWD.est` → `\RPersistBCRhoFWDEst` → 0.59 |
| m3 | Cobertura de intervalos y P(cuartil superior) | `unc.BC.covGauss` → `\RUncBCCovGauss` → 79\%<br>`unc.BC.topGivenTop.obs` → `\RUncBCTopGivenTopObs` → 0.50<br>`unc.BC.topGivenTop.model` → `\RUncBCTopGivenTopModel` → 0.49 |
