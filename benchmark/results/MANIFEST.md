# Manifest: benchmark/results raw runs moved off main

These are the raw runs in this folder that no gate, test, script or document reads or cites by name: ablations, pre-fix and superseded LoCoMo runs, per-question dumps, multilingual screens, load sweeps (item 8 of [ROADMAP-2026-09](../reports/ROADMAP-2026-09.md): retire superseded artifacts). Every file a gate, a test, the Makefile or a report names stays here; the reports that summarise the runs are in [../reports/](../reports/README.md).

The raw files below moved off `main` on 2026-11-15 to keep its tree small; none was lost. They
are on the orphan branch **`data/research-archive`** of this repository, at the same paths
(one commit per folder). To get them back:

```bash
git fetch origin data/research-archive
git restore --source=origin/data/research-archive -- <path>   # one file, or a folder
```

Check a file against this list with `sha256sum <path>`. 97 files, 54,270,797 bytes.

| path | bytes | sha256 |
|---|---|---|
| `benchmark/results/abl_both.json` | 287,123 | `930590c07bade3c6e37d7a32a56b555c75537ea37268b4c1b454a4cc9adcacf8` |
| `benchmark/results/abl_control.json` | 286,966 | `319acac0da93ee20208308697b2e44428de56598caf46700b92caac29c6c0fce` |
| `benchmark/results/abl_share35.json` | 286,811 | `a49abc1e1809a3e49f2200410c860fe34b869ee140f81631548153bb69952c21` |
| `benchmark/results/abl_shipped.json` | 286,599 | `f0ea6dd8cf226b7f55b3f2c19181f29e713eb26112fa2ae07723780edbb38b81` |
| `benchmark/results/abl_subsume.json` | 287,115 | `0c1a839d40612988222163a10cb4c2a0d7e354ad6c7a17eea91ec050a0c045de` |
| `benchmark/results/abl_tail.json` | 286,848 | `86cdab91118919ececcb1f490ed9ba5c1b8b085088a13cc28b971cf6e00588fa` |
| `benchmark/results/brief_reads_20260927.json` | 872 | `b15638df88a36df7b8ad0e7110bc833984f3418b8791d6234f3c28aa9bea08d4` |
| `benchmark/results/capability_validation_20260927.json` | 26,739 | `9cd4ee70fa624b09b45634064c5ab027e8f65b54b0f8ba28c07982c7069f7d88` |
| `benchmark/results/checkpoint_validation_20260928.json` | 631 | `5658596e50203742268e85eeeef9b4d6a26c3fb24431f46f29da9b2bc73bbc87` |
| `benchmark/results/degenerate_fixed.json` | 9,905 | `4f1142dfa0a202d9d379844481614dc5aed8bcbc56b7d6cbe99d6be655acaf14` |
| `benchmark/results/encoder_runtime_devbox_repeat.json` | 1,657 | `7f50c0160d5ba286e03466cff4a9dda5d7953de395128ae07fdf3b2ba23e7fc0` |
| `benchmark/results/final_judged.json` | 403,057 | `63c63521852b8db1912551d25f1099f85087d99cca03c7aaf17a564a98a60c5b` |
| `benchmark/results/final_shipped.json` | 395,508 | `b268f4d089b9e73798fa7954fbdf213857afddf7ac5e973a61b07c19c7ab06e4` |
| `benchmark/results/flag_off_abstain_when_insufficient.json` | 32,646 | `e241def5a448b06a52482adafb1869260ed178504f41759f331f29f74d2b22b5` |
| `benchmark/results/flag_off_bm25.json` | 32,329 | `6a5ecb6444d4cc2436a4d0375613299ab286d8ce012ca0dae32aa8eed66735a1` |
| `benchmark/results/flag_off_contextual_chunks.json` | 32,626 | `637ab1fed5002a48a78c879438444306a484e0b97cbc7ce40660f568e9ab21e4` |
| `benchmark/results/flag_off_definition_expansion.json` | 32,617 | `2596ea3053d382b450aaa4982158830da8fb4a648d086b63fab645a4b548c027` |
| `benchmark/results/flag_off_dense.json` | 33,087 | `de15f39648505c8aac9dc5f6ca596249d6cf9433718afdd028e8b03a4ecdf68a` |
| `benchmark/results/flag_off_evidence_verification.json` | 32,627 | `83fa001033d40541c3634c314f40dd7eab1957cedb26e5fe393f46e72ab116a1` |
| `benchmark/results/flag_off_exact.json` | 32,622 | `c8678fa9d0628c9b14fbfba2942e11dff6d53bd6863294915b4574f7ad76566b` |
| `benchmark/results/flag_off_neighbor_expansion.json` | 32,627 | `8b552b55f6a1b1a20ceb2208f52bb5b6d97a61b0065dc29d8649831ce5ab52c3` |
| `benchmark/results/flag_off_parent_expansion.json` | 32,606 | `20318ade3ab591b912abe88c5a3b6531650e9c1ca9306c7d49b218980d52bb62` |
| `benchmark/results/full_nollm_shipped.json` | 2,142,368 | `c690960b5e0ea84b16b6df7e38b2326ef8de8db3c865e392ba7844d15547904c` |
| `benchmark/results/golden_off_contextual_chunks.json` | 10,297 | `e2153489e90623354405f38c0d794fe013821d3a793782e441610f88ec3c8a27` |
| `benchmark/results/golden_off_evidence_verification.json` | 10,387 | `d8684c34627c30a986077823ffb1435e5ba56d35470e880525e6b31165d6f047` |
| `benchmark/results/golden_off_graph.json` | 10,283 | `0a07a7a2e65c2479821a4a3ea1dc7f7a4a7177a3972a5fc2ac2a14ec82dabd53` |
| `benchmark/results/golden_off_neighbor_expansion.json` | 10,325 | `79ad5eae77b624996d39d8ad9da3f7bd21f45e139a9e4c38c2c0d5b310184a08` |
| `benchmark/results/golden_off_parent_expansion.json` | 10,255 | `05499c5406762a6dc65e178bb8539219fbde6bfdd087d138378ba5f90d2fba83` |
| `benchmark/results/load_runtime_4core_10rps_cold.json` | 4,814 | `e989d391bc1cde0eb5953c4118b1264eccc8e963526f0c512de92db05c78926d` |
| `benchmark/results/load_runtime_4core_30rps_cold.json` | 4,829 | `4dad257ec8c749caa348b64de4652c4042f341253f8b731b93ff0861473e5daf` |
| `benchmark/results/load_runtime_4core_4rps_cold.json` | 4,800 | `8beba52424f175ac81d453ca0fa7ba611621cf838621cafeec3b140c5777e1ef` |
| `benchmark/results/load_runtime_4core_60rps_cold.json` | 5,658 | `2b1ea5fc6070d2defe97e8b1dce105227b2c6ea37b2164b8040195143e9a9de3` |
| `benchmark/results/local_narrative_qwen35_08b.json` | 23,334 | `1643740e9431d5d1496a96bed53951c231ff7badcb7c1a9a9a5fc7438b9942ef` |
| `benchmark/results/locomo_clean_off.json` | 71,111 | `1389f6371b8dbde63cee357e8febf2988aa36f92412bea3121bbf9d2bdfc3420` |
| `benchmark/results/locomo_clean_on.json` | 71,080 | `de37b4d59bc80566171318c4acd618fb86121204dcb842064d68dd25a3f19049` |
| `benchmark/results/locomo_fixed_judged.json` | 381,602 | `040dc15feb89c7c21f12fc5e0a3765f96b6fc00edfae941dc4750cfbe52a4ac1` |
| `benchmark/results/locomo_fixed_shipped.json` | 331,585 | `c4dcfb26e70df6e7cc173525ef95c42abef578ab7a82b603a668023e037b04ea` |
| `benchmark/results/locomo_judge_baseline.json` | 8,190 | `26272d6b2b894719cb756e62d8643d9f42f9f99045273e8ffcc180172945f72e` |
| `benchmark/results/locomo_judge_smoke.json` | 9,216 | `17fe75f78adf83c81c735500dba7c82a3206345454a851eacf50ecf3187bc99b` |
| `benchmark/results/locomo_judged_fixed.json` | 207,528 | `58611ffe4524ec6800a28731601017fdf9bb2b876869297559f2a3385c472843` |
| `benchmark/results/locomo_judged_full.json` | 2,460,510 | `9861dc979a9409decd67265dafe657383caaa643a6d2915afbe890f7c01ba7d0` |
| `benchmark/results/locomo_judged_onnx.json` | 389,867 | `ef705b483b9e7b53f23cfd701d08ce8f6f72cc337c3f67b73632f9399897d888` |
| `benchmark/results/locomo_judged_rerank_on.json` | 62,638 | `586c1d9e9b25fbdb3b9c5f0d537dd99cbe2fb26a5c0c42c3e197e21cae824998` |
| `benchmark/results/locomo_judged_v2_lenient.json` | 267,553 | `0995e224d51efaf3c754f3329b21b0154c36284b07d2de2fb0966277be6146bd` |
| `benchmark/results/locomo_judged_v5_lenient.json` | 378,633 | `7b4eecb40bf5bcf0e6dcb1311158e2cb1edc3e43229ceacc9ddd43ffe2edc26a` |
| `benchmark/results/locomo_judged_v5_refined.json` | 383,957 | `0cd9533dc626cbd167be12e41a574d7a96b601d09a0ee8ceeba5d6fb5447fcf8` |
| `benchmark/results/locomo_judged_verbatim.json` | 329,236 | `f64dc62d64c2ad84d90a7aa83ca6a8e1be753c3eeea96c9b2984eb429191b778` |
| `benchmark/results/locomo_rerank_off.json` | 68,425 | `193a40150bdf911ce453a5fe5f4609b3ea3f06033c45b1d9ac3ccd51db3bacc8` |
| `benchmark/results/locomo_rerank_on.json` | 68,403 | `f8c754c75823d95bc581d5ff75b226aec9f1f2c74379f5d0926d8b843520eb2b` |
| `benchmark/results/locomo_smoke.json` | 8,965 | `65d37ce8ec3e71bcaf1565f0821580839b20ffc0fc19e8aa3687691019b534e9` |
| `benchmark/results/locomo_source_comparison_20260927.json` | 66,807 | `1741d8e9c72ae754c8a375fb28423a49352db2be046bec85042f41620e56ddbb` |
| `benchmark/results/locomo_source_failure_buckets_20260927.json` | 4,095 | `f5ad490de9bb8d189d00fd9087190fb25251e4b0adc961835b25e98be91ddc39` |
| `benchmark/results/locomo_threaded_off.json` | 378,193 | `68bb2c631b097dfcaf6f03b69220ff9551af767c85c71cbcba6a1caac72966cc` |
| `benchmark/results/locomo_threaded_on.json` | 388,603 | `81b3c65678969d0c5a27bd5259139a7cc246bbad4011b1f403199ccfbc5de0ed` |
| `benchmark/results/locomo_v5_failures.json` | 19,839 | `40e02592565c70f709e90f23d1570c1ec997628d48c67f6b0ec0d6699e9bfbb7` |
| `benchmark/results/locomo_verbatim_on.json` | 172,884 | `36be9eee53eaab2b6c443ed09ee16e6ad5e38be4cbe1f6e70bf25223269e6ba1` |
| `benchmark/results/multilingual_dense_bekko_a25m.json` | 5,334,717 | `10aaf13571355282c3dd38dd35bc0a7cf266fa08019db77f3b31d484947c2e49` |
| `benchmark/results/multilingual_input_budget_audit.json` | 4,424 | `1ddac3000a7476c1112e2daaab21b22398a93a8e2f70eb5a30fd76137bda8480` |
| `benchmark/results/multilingual_input_budget_bounded.json` | 4,500 | `4e5ccd38e4af00e43d1b9d7728294b20738f1ddc43b9a315b6c5b6d0aed715df` |
| `benchmark/results/multilingual_nli_mdeberta.json` | 955,934 | `00232426ddbc1d0b947577682fa54523903d0771d62d8b5653fc293acc515a7e` |
| `benchmark/results/multilingual_nli_mdeberta_fp32_diagnostic.json` | 63,185 | `ad355a3498cefe82273d3f0ce1896939bdcc67b027e1b44f257f10b3faf00019` |
| `benchmark/results/multilingual_ocr_tesseract_image_failure.json` | 33,235 | `67d20768afa485e56ac5c9a32d20b745119393da36b75397a8964cc1cb9af92f` |
| `benchmark/results/multilingual_validation_20260927.json` | 24,383 | `862dcfbcd3229bf4e21bb6565e254b673adab5c0ae471f6a723b30282cd79ef4` |
| `benchmark/results/p99_shipped_clean.json` | 401,350 | `5da89948b384e0929b9316f16f4369899ba9d724cf9388ffd08ac53621b5133e` |
| `benchmark/results/scifact1000_off.json` | 32,620 | `0118933592442c9f12139c5c43bbf6e956e0770e121be9a45ee6fa0344fdc6fc` |
| `benchmark/results/scifact1000_on.json` | 33,127 | `480185eb2463bbccd6440a1324808bf9442911787285096ab8e1045d730c37b1` |
| `benchmark/results/scifact300_off.json` | 11,619 | `84a52f2908b01d1f4dc738b10f3bce4b5aa1fef0691df26a5246aaeee843b184` |
| `benchmark/results/scifact300_on.json` | 11,768 | `3896c05288f1bb168e8e8577d8a2353501d9c44b2b5ed1299b0b95dfcee1b8bb` |
| `benchmark/results/scifact600_off.json` | 16,407 | `3998ec511c280f34d8079088f4217eec1761fdeeb8dd4fcb3414b46dff3124a2` |
| `benchmark/results/scifact600_on.json` | 16,476 | `8eb07e52af4008beee0b73ef88887fb008f067aa3a02ede97076a03d79065d85` |
| `benchmark/results/scifact600_rerank_off.json` | 3,337 | `42d5801e71ed7c3a8cd66e176de9788c16b218c623e2621c8a1d0eab2ad3de2b` |
| `benchmark/results/scifact600_rerank_on.json` | 3,320 | `5cae92ffa47175cdecae9709da22810eba3793d864a951425fac7ae3834e9825` |
| `benchmark/results/scifact_dense_bekko_a25m.json` | 617,849 | `3f5a14ad5dae728f6c4c7d9e66f08d943d3a0d0d4f5c6ddf5248cdaceed7e470` |
| `benchmark/results/scifact_dense_comparison_20260927.json` | 1,296 | `a0feb8b6fe9158c6ce419ddbee42e95de2ae17cbe78c5d2977152942450bb0df` |
| `benchmark/results/scifact_dense_e5.json` | 617,438 | `d408393dbe7f04a1ada912f9b5439ebd568be9985da19fcd840c3f088ac23f09` |
| `benchmark/results/scifact_dense_granite_multilingual.json` | 617,596 | `d1cee22223068d42bb5f573b987fab6710c748815724aebf4069dcb002aca041` |
| `benchmark/results/scifact_fusion_paired_screen.json` | 1,354 | `cd001e4d65007b56d641c19bf16634d8b3f63c684b522566034860b7583ac731` |
| `benchmark/results/scifact_granite_e5_fusion_screen.json` | 59,054 | `95ec1193e2ddd7f0c168475f220f4b91f169d7655b259653e41c812a226417a1` |
| `benchmark/results/scifact_title_weight_screen.json` | 294,035 | `e36336485873c5506df177318373805876f82f01f0db81ea669ec0bf41387657` |
| `benchmark/results/step0_head30_group_collapse.json` | 2,219,731 | `d51c7910a78095b17130759dae0247069753801f863ba732b8583180cfd244b4` |
| `benchmark/results/sweep_k200_arms.json` | 3,344,593 | `90a3d36031fe6e4e27a2d81a0bba1f0a5fb95d04b3023bcc5da044122e00f7a2` |
| `benchmark/results/v2_judged_shipped.json` | 4,539,399 | `4b1d25a906374edf4eb707d29325e2bb2db9304813ff4b26ca1aae7b0c7fab9c` |
| `benchmark/results/v2_nollm_shipped.json` | 3,819,708 | `4ec6e6b2e70a3910d29130b742773d2c92542ecd06fb603f996dd734b281d08f` |
| `benchmark/results/warm_discard.json` | 286,930 | `f8b144ca39c4c5f404534c31192182edaca73b311f88206750f4127a1e912861` |
| `benchmark/results/arms/arm_a.json` | 3,816,181 | `f92001c2a03c67e0df804ec145d29772733194f48ddad708d2330ec26dcceb60` |
| `benchmark/results/arms/arm_b.json` | 3,816,049 | `8908ff85be5ee47f90bc1eae6a5c8043b43489ac8f700935153b139aca7152eb` |
| `benchmark/results/arms/arm_base.json` | 3,816,412 | `d5d6342bf2be14adbcbeabee351a820962d084ef48081e43d7e4f4220b639198` |
| `benchmark/results/arms/arm_c.json` | 3,854,473 | `8b2ddc27ebe0852fbbe68d76225b93467c05e19b85684e913473d16486648c06` |
| `benchmark/results/arms/arm_d.json` | 3,815,574 | `c42f15c625e288e0f50dd481663ab8d41818dc46e420905aaf94283de5a5e038` |
| `benchmark/results/overhaul/context_latency_after.json` | 6,116 | `d88979f9a5315bd32fdd7bb41986388838029165e7bfe0d3dbe2ba20fa04db6c` |
| `benchmark/results/overhaul/context_latency_before.json` | 5,653 | `cd68b232963d57b8fa5010958544fd7ba257bf51894c18c273560c23ee767ffb` |
| `benchmark/results/overhaul/context_latency_final.json` | 6,120 | `eb727e57bf0a19d653a886c9b619cef8b661a5a517663d16852bd13442c4f07f` |
| `benchmark/results/overhaul/context_latency_floor0.json` | 6,123 | `6246aad589195818f5995d13c5c99f97ee8629be9b066a39df180d8edf579c12` |
| `benchmark/results/overhaul/locomo_source_consolidation_off.summary.json` | 34,758 | `ba067f8c22f3437080bef07a861750cc39576c0e473380129ee6d4006d2b7fc0` |
| `benchmark/results/overhaul/locomo_source_consolidation_on.summary.json` | 34,739 | `1b98f86022cda9c7b8dfa58114b7b21cf738e3a3f7bb758e3ee973c555c8b7ad` |
| `benchmark/results/overhaul/locomo_source_control.summary.json` | 35,064 | `8a7c1e2bd515207b2d00287699ee974ebbcd275377be3ceac712c2b3ba5cf3e2` |
| `benchmark/results/overhaul/locomo_source_entity_search.summary.json` | 34,262 | `6c6da035fee15adcd3061239b5e8e42517f39b4345cd8309c47cd93152e184bf` |
