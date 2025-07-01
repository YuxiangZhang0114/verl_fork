set -x

# Example running script for RFT.
# Paths to your datasets
train_path=$HOME/data/train.parquet
test_path=$HOME/data/test.parquet

train_files="['$train_path']"
test_files="['$test_path']"

model_path=YourModel/Path

python3 -m recipe.RFT.main_rft \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    actor_rollout_ref.model.path=$model_path \
    trainer.total_epochs=1 "$@"
