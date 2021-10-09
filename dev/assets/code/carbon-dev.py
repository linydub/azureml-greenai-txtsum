# td: tracker decorator
# scope (entire_run/total, finetune, evaluate, predict)
# track_carbon arg(bool)
class MetricTracking:
    def __init__(self, func):
        self._enabled = False
        self._func = func

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, new_value):
        if not isinstance(new_value, bool):
            raise ValueError("enabled can only be set to a boolean value")
        self._enabled = new_value

    def __call__(self, target):
        if self._enabled:
            return self._func(target)
        return target

carbon_tracker = MetricTracking(track_emissions)
#carbon_tracker.enabled = args.track_carbon

# tracking_scope(finetune)
@track_emissions(project_name="carbon-finetune", output_dir=training_args.output_dir)
def finetune(trainer, checkpoint):
    return trainer.train(resume_from_checkpoint=checkpoint)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = finetune(trainer, checkpoint) # edit
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
