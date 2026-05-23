#marker
python judge_results7.py --real_source marker_training_dataset_384_MPP09_stratified.parquet --gen_folder /net/pr2/projects/plgrid/plgghistpat/artifacts_applied/dataset_marker_ANORAK  --artifact_type marker --clean_source /net/pr2/projects/plgrid/plgghistpat/datasets/nnUNet/raw/Dataset011_ANORAK/imagesTr/  --calc_ssim --output_file my_fidelity_resultsCI.txt
python judge_results7.py --real_source marker_training_dataset_384_MPP09_stratified.parquet --gen_folder marker_test_procedural_gen_full --artifact_type marker --clean_source marker_training_dataset_384_MPP09_stratified.parquet --calc_ssim --output_file my_fidelity_resultsCI.txt

#dust
python judge_results7.py --real_source dust_training_dataset_384_MPP09.parquet --gen_folder /net/pr2/projects/plgrid/plgghistpat/artifacts_applied/dataset_dust_ANORAK --artifact_type dust  --clean_source /net/pr2/projects/plgrid/plgghistpat/datasets/nnUNet/raw/Dataset011_ANORAK/imagesTr/  --calc_ssim --output_file my_fidelity_resultsCI.txt
python judge_results7.py --real_source dust_training_dataset_384_MPP09.parquet --gen_folder dust_on_tcga_clean --artifact_type dust --clean_source dust_training_dataset_384_MPP09.parquet --calc_ssim --output_file my_fidelity_resultsCI.txt

#folds
python judge_results7.py --real_source combined_dataset_folds_and_clean_MPP_09_ttvsplit.parquet --gen_folder /net/pr2/projects/plgrid/plgghistpat/artifacts_applied/dataset_folds_ANORAK --artifact_type folds --clean_source /net/pr2/projects/plgrid/plgghistpat/datasets/nnUNet/raw/Dataset011_ANORAK/imagesTr/  --calc_ssim --output_file my_fidelity_resultsCI.txt
python judge_results7.py --real_source combined_dataset_folds_and_clean_MPP_09_ttvsplit.parquet --gen_folder generated_results_folds_tcga_testset_v2-ch_01_st_09_sc7_5 --artifact_type folds --clean_source combined_dataset_folds_and_clean_MPP_09_ttvsplit.parquet --calc_ssim --output_file my_fidelity_resultsCI.txt

#precitipates
python judge_results7.py --real_source lung_precipitate_training_dataset_MPP_09_384px_split.parquet --gen_folder /net/pr2/projects/plgrid/plgghistpat/artifacts_applied/dataset_precitipates_ANORAK --artifact_type precipitates --clean_source /net/pr2/projects/plgrid/plgghistpat/datasets/nnUNet/raw/Dataset011_ANORAK/imagesTr/  --calc_ssim --output_file my_fidelity_resultsCI.txt
python judge_results7.py --real_source lung_precipitate_training_dataset_MPP_09_384px_split.parquet --gen_folder test_precitipates_ch2500_st09_gs_7_5 --artifact_type precipitates --clean_source lung_precipitate_training_dataset_MPP_09_384px_split.parquet --calc_ssim --output_file my_fidelity_resultsCI.txt