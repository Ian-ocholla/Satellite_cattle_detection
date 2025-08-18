"""
Trains the model
"""
import os
import torch
import logging
import gc
import matplotlib.pyplot as plt
import json
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine.hooks import HookBase
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
import detectron2.utils.comm as comm

class LumoTrainer(DefaultTrainer):
    """
    Custom trainer class with an overridden evaluator.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)
    

class BestModelHook(HookBase):
    """
    Hook to save the best model based on evaluation metric (bbox/AP50).
    """
    def __init__(self, cfg, metric="bbox/AP50", min_max="max"):
        self._period = cfg.TEST.EVAL_PERIOD
        self.metric = metric
        self.min_max = min_max
        self.best_value = float("-inf") if min_max == "max" else float("inf")
        self._logger = logging.getLogger("detectron2")

    #def _take_latest_metrics(self):
    #    with torch.no_grad():
    #        latest_metrics = self.trainer.storage.latest() ## error cause
    #        return latest_metrics

    def after_step(self):
        """
        stop training is validation loss does not improve
        """
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            #latest_metrics = self._take_latest_metrics()
            dataset_name = self.trainer.cfg.DATASETS.TEST[0]
            evaluator = COCOEvaluator(dataset_name, self.trainer.cfg, distributed= False, output_dir=self.trainer.cfg.OUTPUT_DIR)
            val_loader = build_detection_test_loader(self.trainer.cfg, dataset_name)
            results = inference_on_dataset(self.trainer.model, val_loader, evaluator)

            print("==Raw COCO results ==")
            print(json.dumps(results, indent=2))

            cat_key, metric_key = self.metric.split("/")
            value = results.get(cat_key, {}).get(metric_key)

            if value is not None:
                #assert 0.0 <= value <= 1.0, f"{self.metric} is out of range: {value}"
                if (self.min_max == "min" and value < self.best_value) or (self.min_max == "max" and value > self.best_value):
                    self._logger.info(f"Updating best model at iteration {self.trainer.iter} with {self.metric} = {value:.4f}")
                    self.best_value = value
                    self.trainer.checkpointer.save("model_best")
                    self._logger.info(f"Best {self.metric} so far: {self.best_value:.4f}")
            else:
                self._logger.warning(f"Metric {self.metric} not found in evaluation results.")
                

            """if results and "bbox" in results and self.metric.split("/")[-1] in results["bbox"]:
                value = results["bbox"][self.metric.split("/")[-1]]
                print(f"Raw AP50 value from evaluatot: {results['bbox']['AP50']}")

                if (self.min_max == "min" and value < self.best_value) or (self.min_max == "max" and value > self.best_value):
                    self._logger.info(f"Updating best model at iteration {self.trainer.iter} with {self.metric} = {value}")
                    self.best_value = value
                    self.trainer.checkpointer.save("model_best")
        print(f"Raw AP50 value from evaluatot: {results['bbox']['AP50']}")

            #for (key, (value, iter)) in latest_metrics.items():
            #    if key == self.metric:
            #        if (self.min_max == "min" and value < self.best_value) or (self.min_max == "max" and value > self.best_value):
            #            self._logger.info(f"Updating best model at iteration {iter} with {self.metric} = {value}")
            #            self.best_value = value
            #            self.trainer.checkpointer.save("model_best")
            """

class EarlyStoppingHook(HookBase):
    """
    Implements early stopping based on validation loss.
    """
    def __init__(self, patience=1200, min_delta=0.001, metric="val_total_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric  # Track validation loss
        self.best_loss = float("inf")
        self.wait = 0
        self.logger = logging.getLogger("detectron2")

    def after_step(self):
     """
     Stop training if validation loss does not improve.
     """
     latest_metrics = self.trainer.storage.latest()
     if self.metric in latest_metrics:
         current_loss = latest_metrics[self.metric][0]  # Get latest validation loss

         # Check for improvement
         if current_loss < self.best_loss - self.min_delta:
             self.best_loss = current_loss
             self.wait = 0
         else:
             self.wait += 1

     #if self.wait >= self.patience:
     #    print(f"Early stopping triggered at iteration {self.trainer.iter}. Stopping training.")
     #    self.trainer.checkpointer.save("model_early_stop")  # Save the last best model
         
         # Stop training gracefully by setting the iteration count to max_iter
     #    self.trainer.iter = self.trainer.max_iter  

     if self.wait >= self.patience:
         print(f"Early stopping triggered at iteration {self.trainer.iter}. Stopping training.")
         self.trainer.checkpointer.save("model_early_stop")
         self.trainer.storage.put_scalar("early_stop", 1)
         #raise StopIteration("Early stopping triggered.")
         self.trainer.iter = self.trainer.max_iter
         #self.trainer._shutdown = True


class ValLossHook(HookBase):
    """
    Hook to compute validation loss after each step.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        
        self.mapper = DatasetMapper(cfg, is_train=True, augmentations=[T.NoOpTransform()]) #change from true
        self.dataset_name = self.cfg.DATASETS.TEST[0]
        self.reset_loader()

    
    def reset_loader(self):
        self._loader = iter(build_detection_test_loader(self.cfg, dataset_name=self.dataset_name, mapper=self.mapper))

    def after_step(self):
        """
        Computes validation loss and stores it in the trainer storage.
        """
        try:
            data = next(self._loader)
        except StopIteration:
            self.reset_loader()
            data = next(self._loader)

        
        with torch.no_grad():
            #loss_dict = self.trainer.model(data)
            self.trainer.model.train()
            loss_dict = self.trainer.model(data)
            losses = sum(loss_dict.values())

            # Ensure loss is finite
            assert torch.isfinite(losses).all(), loss_dict

            # Reduce across multiple GPUs if needed
            loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss_dict_reduced.values())

            if comm.is_main_process():
                self.trainer.storage.put_scalars(val_total_loss=losses_reduced, **loss_dict_reduced)

from detectron2.data.transforms import ResizeShortestEdge, RandomFlip, RandomRotation, RandomBrightness #, RandomContrast

def build_augmentation():
    
    #Defines the augmentation transformations.
    
    return [
        #RandomFlip(horizontal=False, vertical=False),  # Horizontal Flip
        #RandomFlip(horizontal=False, vertical=False),  # Vertical Flip
        #RandomRotation([0]),  # Only 90-degree rotation
        #RandomBrightness(0.8, 1.2),  # Safe brightness change
        #RandomContrast(0.9, 1.1),  # Safe contrast change..seems to work against rather than for the model
        ResizeShortestEdge(short_edge_length=(800,1024), max_size=1024, sample_style='choice')  # Match dataset size
    ]


def train_model(cfg):
    """
    Runs training with the custom trainer and best model hook.
    """
    trainer = LumoTrainer(cfg)

    patience_iterations = 1200
    early_stopping = EarlyStoppingHook(patience=patience_iterations, metric="val_total_loss")
    
    bm_hook = BestModelHook(cfg, metric="bbox/AP50", min_max="max")
    val_loss_hook = ValLossHook(cfg)  # Register validation loss hook

    trainer.register_hooks([bm_hook, early_stopping, val_loss_hook])

    # Use the correct augmentations
    augmentations = build_augmentation()  # Apply augmentations
    mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)  # Apply augmentations here
    #mapper = DatasetMapper(cfg, is_train=True)
    
    trainer.data_loader = build_detection_train_loader(cfg, mapper=mapper)  # Use custom loader with augmentations

    #trainer.data_loader = build_detection_train_loader(cfg)  # Use custom loader
    trainer.train()

    # Trigger garbage collection
    gc.collect()
            
    return trainer