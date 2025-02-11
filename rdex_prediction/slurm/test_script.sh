model_call(){
    python3 run_model.py $1 $2 \
     --dataset="../../data/04_model_input/rdex_prediction_roi_dataset.pkl" \
     --scopes="../../data/04_model_input/rdex_prediction_roi_scopes.pkl" \
     --append="roi"
    #  --n_cores=$SLURM_CPUS_PER_TASK
}

cd ../pipelines/
for model in ridge lasso elastic; do
    for condition in correct_go incorrect_go correct_stop incorrect stop; do
        model_call $model $condition &
    done
done
wait
