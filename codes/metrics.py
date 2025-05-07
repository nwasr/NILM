import tensorflow as tf

class EnergyBasedF1(tf.keras.metrics.Metric):
    """
    Energy-based F1 score for NILM models.
    """
    def __init__(self, rescaling=None, min_value=0.0, max_value=1.0, 
                 mean_value=0.0, std_value=1.0, name='energy_based_f1', **kwargs):
        super(EnergyBasedF1, self).__init__(name=name, **kwargs)
        self.pr_num = self.add_weight(name='pr_num', initializer='zeros')
        self.p_den = self.add_weight(name='p_den', initializer='zeros')
        self.r_den = self.add_weight(name='r_den', initializer='zeros')
        self.rescaling = rescaling
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.std_value = std_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Rescale data if required
        if self.rescaling == 'standardize':
            y_pred_rescaled = y_pred * self.std_value + self.mean_value
            y_true_rescaled = y_true * self.std_value + self.mean_value
        elif self.rescaling == 'normalize':
            y_pred_rescaled = y_pred * (self.max_value - self.min_value) + self.min_value
            y_true_rescaled = y_true * (self.max_value - self.min_value) + self.min_value
        else:
            y_pred_rescaled = y_pred
            y_true_rescaled = y_true

        # Ensure float32 dtype
        y_pred_rescaled = tf.cast(y_pred_rescaled, tf.float32)
        y_true_rescaled = tf.cast(y_true_rescaled, tf.float32)

        # Remove negative values
        y_pred_positive = tf.where(y_pred_rescaled < 0.0, tf.zeros_like(y_pred_rescaled), y_pred_rescaled)

        self.p_den.assign_add(tf.reduce_sum(y_pred_positive))
        self.r_den.assign_add(tf.reduce_sum(y_true_rescaled))
        self.pr_num.assign_add(tf.reduce_sum(tf.minimum(y_pred_positive, y_true_rescaled)))

    def result(self):
        epsilon = 1e-8
        precision = self.pr_num / (self.p_den + epsilon)
        recall = self.pr_num / (self.r_den + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        return f1

    def reset_states(self):
        self.pr_num.assign(0.)
        self.p_den.assign(0.)
        self.r_den.assign(0.)

def compute_F1_score(predicted_values, ground_truth):
    """
    Compute Energy-Based F1 score (static version for arrays).
    """
    pr_num = 0.0
    p_den = 0.0
    r_den = 0.0
    epsilon = 1e-8
    for i in range(len(ground_truth)):  
        p_den += predicted_values[i]
        r_den += ground_truth[i]
        pr_num += min(predicted_values[i], ground_truth[i])
    precision = pr_num / (p_den + epsilon)
    recall = pr_num / (r_den + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return f1
