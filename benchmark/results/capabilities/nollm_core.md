# Capability eval — nollm (2026-10-18)

Dataset `3a7401f20fb1`: 105 scenarios, 111 queries, 1110 phrasings per language. Selection: `{"languages": ["en", "de", "es", "ar", "hi", "mixed"], "split": "all", "capabilities": ["single_fact_recall", "multi_hop", "temporal_recency", "update_contradiction", "abstention", "forgetting_decay", "procedural_skill", "test_time_learning", "implicit_personalization", "explicit_personalization"], "domains": "all", "kinds": "all", "scenarios_per_capability": null, "scenarios_per_domain": null, "scenario_ids": null}`, sample `core`.

**Primary (context/new_session) macro accuracy 0.48**, retail track 0.48, the eleven origin cases 0.46, partial credit 0.74. Context latency p50/p95 45.83/67.39 ms.

| # | Capability | context/new_session | context/same_session | search/new_session | partial | retail | p95 ms |
|---|---|---|---|---|---|---|---|
| 1 | single_fact_recall | 0.99 | 0.99 | 1.00 | 0.99 | 1.00 | 67.51 |
| 2 | multi_hop | 0.49 | 0.70 | 0.60 | 0.71 | 0.46 | 66.24 |
| 3 | temporal_recency | 0.01 | 0.39 | 0.02 | 0.57 | 0.00 | 64.77 |
| 4 | update_contradiction | 0.05 | 0.05 | 0.00 | 0.91 | 0.01 | 67.93 |
| 5 | abstention | 0.32 | 0.15 | - | - | 0.38 | 67.65 |
| 6 | forgetting_decay | 0.46 | 0.46 | 0.50 | 0.46 | 0.45 | 64.19 |
| 7 | procedural_skill | 0.60 | 0.96 | 0.50 | 0.79 | 0.50 | 75.08 |
| 8 | test_time_learning | 0.94 | 0.94 | 0.21 | 0.94 | 0.91 | 68.13 |
| 9 | implicit_personalization | 0.68 | 0.68 | 1.00 | 0.68 | 0.77 | 65.57 |
| 10 | explicit_personalization | 0.27 | 0.97 | 0.27 | 0.62 | 0.28 | 61.36 |

Per language (primary, macro): ar 0.48, de 0.49, en 0.49, es 0.50, hi 0.45, mixed 0.50

Per domain (primary, macro): field_service 0.36, finance_hr 0.53, it_support 0.35, personal 0.71, retail 0.48

Per split (primary, macro): dev 0.51, heldout 0.45

Abstention: precision 0.08, recall 0.23, false-refusal rate 0.30. Stale-value leak rate 0.37 (754 of 2016).

Ingestion: 2712 turns, 3760 jobs (0 failed), written in 232.6 s; settled 244.83 s after the last write. Worker queues at the start: all empty; enrichment still queued when the questions started: all empty.

## Top failure reasons

**abstention** — 345/504 failed: no_abstain 345
- `abstain-field-hilltop-serial/q1/direct/en` (no_abstain): “What's the serial number of the pump at Hilltop?” — missing -, leaked -; got: memory: Hilltop station has one older PX-250 pump.; memory: Riverside station runs two PX-400 pumps, serial numbers R-1182 and R-1183.
- `abstain-finance-cfo-salary/q1/direct/en` (no_abstain): “What is Helen Park's salary?” — missing -, leaked -; got: memory: Our CFO is Helen Park; she joined in January.; memory: Executive compensation is handled by the board's remuneration committee.

**explicit_personalization** — 404/554 failed: missing_evidence 230, false_abstain 174
- `explicit-field-no-north-fridays/q1/direct/en` (missing_evidence): “Which jobs can I take this Friday?” — missing ['J-12'], leaked -; got: profile: - Never assign me jobs north of the river on Fridays — I pick up my kids at 3.; memory: Never assign me jobs north of the river on Fridays — I pick up my kids at 3.
- `explicit-finance-eur-only/q1/direct/en` (false_abstain): “What was our Q2 revenue?” — missing ['4.2'], leaked -; got: profile: - Always report our figures in EUR, never in USD.

**forgetting_decay** — 272/504 failed: false_abstain 144, missing_evidence 128
- `forget-field-temp-generator/q1/direct/en` (missing_evidence): “What's the status of the power equipment currently at Site 4?” — missing ['UPS-4'], leaked -; got: memory: Mains power is restored at Site 4 and the temporary generator GEN-T1 has been removed.
- `forget-it-decommissioned-db/q1/direct/en` (false_abstain): “Give me a health summary of our active database servers.” — missing ['db-01', 'db-03'], leaked -; got: memory: legacy-db-02 has been decommissioned and wiped; remove it from all reports.

**implicit_personalization** — 159/504 failed: false_abstain 159
- `implicit-field-medical/q1/direct/en` (false_abstain): “Write the checklist for tomorrow's job.” — missing ['infection-control | IEC 62353'], leaked -; got: nothing
- `implicit-finance-germany/q1/direct/en` (false_abstain): “Draft a payment reminder for an overdue invoice.” — missing ['EUR | HGB | German'], leaked -; got: nothing

**multi_hop** — 257/504 failed: missing_evidence 157, false_abstain 100
- `multihop-it-payroll-reboot/q1/direct/en` (missing_evidence): “Will payroll be down this weekend?” — missing ['02:00'], leaked -; got: memory: The payroll service runs on host app-07.
- `multihop-retail-aerosol-cage/q1/direct/en` (missing_evidence): “Where should the ColorMax delivery be stored?” — missing ['ColorMax'], leaked -; got: memory: Safety rule: every aerosol product is classed as hazmat and has to be stored in Cage H2.

**procedural_skill** — 203/504 failed: missing_evidence 135, false_abstain 68
- `procedure-field-service-report/q1/direct/en` (missing_evidence): “Write the service report for WO-4120.” — missing ['WO-4120'], leaked -; got: profile: - Every service report I ask for must end with a parts-used list and a customer signature line.; memory: Every service report I ask for must end with a parts-used list and a customer signature line.
- `procedure-retail-hazmat-report/q1/direct/en` (missing_evidence): “Write up an incident report for the leaking bleach drum in aisle 3.” — missing ['UN1791'], leaked -; got: profile: - For any hazmat incident report, always start with the UN number and the SDS reference, then the location, then the action taken.; memory: For any hazmat incident report, always start with the UN number and the SDS reference, then the location, then the action taken.

**single_fact_recall** — 4/504 failed: false_abstain 3, missing_evidence 1
- `recall-finance-cost-centre/q1/implicit/hi` (false_abstain): “लिस्बन की रसीदें, जमा करने को तैयार।” — missing ['CC-4471'], leaked -; got: nothing
- `recall-personal-locker/q1/implicit/hi` (false_abstain): “फिर से लॉकर 317 नहीं खुल रहा।” — missing ['22-08-41'], leaked -; got: nothing

**temporal_recency** — 497/504 failed: stale_leak 281, false_abstain 209, missing_evidence 7
- `temporal-field-compressor/q1/direct/en` (false_abstain): “Is the compressor at Site 12 working?” — missing ['replaced'], leaked -; got: nothing
- `temporal-finance-leave/q1/direct/en` (stale_leak): “Is Maria Lopez available for the audit kickoff?” — missing -, leaked ['until further notice']; got: memory: Maria Lopez is back from parental leave as of Monday and taking her old desk again.; memory: Maria Lopez from accounts payable is on parental leave until further notice.

**test_time_learning** — 32/504 failed: false_abstain 32
- `correction-retail-blind-recount/q1/indirect/en` (false_abstain): “The batteries count came out way off.” — missing ['blind recount'], leaked -; got: nothing
- `correction-retail-short-shipment/q1/implicit/en` (false_abstain): “Six cases short.” — missing ['OS&D'], leaked -; got: nothing

**update_contradiction** — 481/504 failed: stale_leak 438, false_abstain 25, missing_evidence 18
- `update-field-van-assignment/q1/direct/en` (stale_leak): “Who drives van #7?” — missing -, leaked ['Ana Silva']; got: profile: - Van #7 has been reassigned to Peter Brandt; Ana Silva drives van #9 from now on.; memory: Van #7 has been reassigned to Peter Brandt; Ana Silva drives van #9 from now on.
- `update-finance-approval-limit/q1/direct/en` (stale_leak): “What's the team lead approval limit for expenses?” — missing -, leaked ['$500']; got: memory: Policy update: the team lead expense approval limit was raised to $1,000.; memory: Team leads can approve expenses up to $500.

## Failure reasons on context/same_session

**abstention** — 429/504 failed: no_abstain 429
- `abstain-field-hilltop-serial/q1/direct/en` (no_abstain): “What's the serial number of the pump at Hilltop?” — missing -, leaked -; got: memory: Hilltop station has one older PX-250 pump.; memory: Riverside station runs two PX-400 pumps, serial numbers R-1182 and R-1183.
**explicit_personalization** — 18/554 failed: missing_evidence 14, false_abstain 4
- `explicit-retail-no-clearance/q1/direct/ar` (missing_evidence): “ماذا أعيد طلبه لممر الألعاب؟” — missing ['اقتراحات إعادة طلب'], leaked -; got: memory: ممر الألعاب منخفض المخزون: TY-14 مكعب الألغاز تبقّى 3 (تصفية)؛ RG-20 مكعبات بناء تبقّى 4؛ RG-31 دب محشو تبقّى 2.
**forgetting_decay** — 272/504 failed: false_abstain 144, missing_evidence 128
- `forget-field-temp-generator/q1/direct/en` (missing_evidence): “What's the status of the power equipment currently at Site 4?” — missing ['UPS-4'], leaked -; got: memory: Mains power is restored at Site 4 and the temporary generator GEN-T1 has been removed.
**implicit_personalization** — 159/504 failed: false_abstain 159
- `implicit-field-medical/q1/direct/en` (false_abstain): “Write the checklist for tomorrow's job.” — missing ['infection-control | IEC 62353'], leaked -; got: nothing
**multi_hop** — 153/504 failed: missing_evidence 134, false_abstain 19
- `multihop-retail-po-supplier/q1/direct/en` (missing_evidence): “Who should I chase about PO-77812?” — missing ['PackRight'], leaked -; got: memory: I raised PO-77812 for 2,000 corrugated boxes; it's late.; memory: PO-77790 for shrink wrap arrived complete.
**procedural_skill** — 21/504 failed: missing_evidence 11, false_abstain 10
- `procedure-personal-metric-recipes/q1/indirect/en` (false_abstain): “The kids want pancakes for breakfast tomorrow.” — missing ['metric'], leaked -; got: nothing
**single_fact_recall** — 4/504 failed: false_abstain 3, missing_evidence 1
- `recall-finance-cost-centre/q1/implicit/hi` (false_abstain): “लिस्बन की रसीदें, जमा करने को तैयार।” — missing ['CC-4471'], leaked -; got: nothing
**temporal_recency** — 308/504 failed: stale_leak 281, false_abstain 20, missing_evidence 7
- `temporal-finance-leave/q1/direct/en` (stale_leak): “Is Maria Lopez available for the audit kickoff?” — missing -, leaked ['until further notice']; got: memory: Maria Lopez is back from parental leave as of Monday and taking her old desk again.; memory: Maria Lopez from accounts payable is on parental leave until further notice.
**test_time_learning** — 32/504 failed: false_abstain 32
- `correction-retail-blind-recount/q1/indirect/en` (false_abstain): “The batteries count came out way off.” — missing ['blind recount'], leaked -; got: nothing
**update_contradiction** — 481/504 failed: stale_leak 438, false_abstain 25, missing_evidence 18
- `update-field-van-assignment/q1/direct/en` (stale_leak): “Who drives van #7?” — missing -, leaked ['Ana Silva']; got: profile: - Van #7 has been reassigned to Peter Brandt; Ana Silva drives van #9 from now on.; memory: Van #7 has been reassigned to Peter Brandt; Ana Silva drives van #9 from now on.

## Failure reasons on search/new_session

**explicit_personalization** — 404/554 failed: missing_evidence 404
- `explicit-field-no-north-fridays/q1/direct/en` (missing_evidence): “Which jobs can I take this Friday?” — missing ['J-12'], leaked -; got: memory: Never assign me jobs north of the river on Fridays — I pick up my kids at 3.; message: USER: Never assign me jobs north of the river on Fridays — I pick up my kids at 3.
**forgetting_decay** — 254/504 failed: missing_evidence 254
- `forget-field-temp-generator/q1/direct/en` (missing_evidence): “What's the status of the power equipment currently at Site 4?” — missing ['UPS-4'], leaked -; got: memory: Mains power is restored at Site 4 and the temporary generator GEN-T1 has been removed.; message: USER: Mains power is restored at Site 4 and the temporary generator GEN-T1 has been removed.
**multi_hop** — 200/504 failed: missing_evidence 200
- `multihop-it-payroll-reboot/q1/direct/en` (missing_evidence): “Will payroll be down this weekend?” — missing ['02:00'], leaked -; got: memory: The payroll service runs on host app-07.; message: USER: The payroll service runs on host app-07.
**procedural_skill** — 250/504 failed: missing_evidence 200, not_retrieved 50
- `procedure-field-service-report/q1/direct/en` (missing_evidence): “Write the service report for WO-4120.” — missing ['WO-4120'], leaked -; got: memory: Every service report I ask for must end with a parts-used list and a customer signature line.; message: USER: Every service report I ask for must end with a parts-used list and a customer signature line.
**temporal_recency** — 494/504 failed: stale_leak 294, not_retrieved 200
- `temporal-field-compressor/q1/direct/en` (not_retrieved): “Is the compressor at Site 12 working?” — missing ['replaced'], leaked -; got: nothing
**test_time_learning** — 396/504 failed: stale_leak 396
- `correction-field-flange-torque/q1/direct/en` (stale_leak): “What torque for the PX-400 flange bolts?” — missing -, leaked ['40 Nm']; got: memory: That's wrong for the PX-400 — its flange bolts go to 55 Nm in a star pattern.; message: USER: What torque do the flange bolts on the PX-400 take?
**update_contradiction** — 504/504 failed: stale_leak 504
- `update-field-van-assignment/q1/direct/en` (stale_leak): “Who drives van #7?” — missing -, leaked ['Ana Silva']; got: memory: Van #7 has been reassigned to Peter Brandt; Ana Silva drives van #9 from now on.; message: USER: Van #7 has been reassigned to Peter Brandt; Ana Silva drives van #9 from now on.

Stale-value leak rate by surface: context/new_session 0.37 (754/2016), context/same_session 0.37 (754/2016), search/new_session 0.59 (1194/2016)

