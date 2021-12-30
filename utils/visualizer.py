# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import tensorflow as tf


class Visualizer():
    def __init__(self, args):
        self.args = args
        self.tf = tf
        self.log_dir = os.path.join(args.checkpoint_path, 'logs')
        self.writer = tf.summary.FileWriter(self.log_dir)
    
    def plot_loss(self, loss, step, tag):
        summary = self.tf.Summary(
            value=[self.tf.Summary.Value(tag=tag, simple_value=loss)])
        self.writer.add_summary(summary, step)

    def plot_dict(self, loss, step):
        for tag, value in loss.items():
            summary = self.tf.Summary(
                value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)
