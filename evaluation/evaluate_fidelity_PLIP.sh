#folds
python judge_results_PLIP.py --gen_folder /net/pr2/projects/plgrid/plgghistpat/artifacts_applied/dataset_folds_ANORAK --artifact_type folds 
python judge_results_PLIP.py --gen_folder generated_results_folds_tcga_testset_v2-ch_01_st_09_sc7_5 --artifact_type folds

#precitipates
python judge_results_PLIP.py  --gen_folder /net/pr2/projects/plgrid/plgghistpat/artifacts_applied/dataset_precitipates_ANORAK --artifact_type precipitates
python judge_results_PLIP.py  --gen_folder test_precitipates_ch2500_st09_gs_7_5 --artifact_type precipitates