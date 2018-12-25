import tensorflow as tf

optimizer = tf.train.AdamOptimizer(1e-4)

def normalize_loss(loss):
    return loss / (tf.abs(tf.stop_gradient(loss)) + 0.5)

@tf.contrib.eager.defun
def train_func(model,boards,actions,probabilities,advantages,values,masks):

    with tf.GradientTape() as tape:
                
        policy, value = model(boards)
        
        total_loss = 0

        value = value*masks
        
        value_loss = tf.nn.l2_loss(value-values)
        
        responsible_outputs = tf.reduce_sum(actions*policy,3)
        
        ratios = (responsible_outputs + 1e-8)/(probabilities + 1e-8)
        policy_loss = -tf.reduce_sum(tf.minimum(
                                    tf.clip_by_value(ratios, 1/200, 200) * advantages,
                                    tf.clip_by_value(ratios, 1.0 - 0.1, 1.0 + 0.1)*advantages))
        
        entropy_loss = tf.reduce_sum(policy * tf.log(policy + 1e-12)*masks)
        
        regularization = tf.reduce_sum([tf.reduce_sum(tf.abs(x)) for x in model.variables])
        
        total_loss = value_loss+policy_loss+0.0002*regularization + 0.0001*entropy_loss
        
        gradients,norm = tf.clip_by_global_norm(tape.gradient(total_loss,model.variables),2000)
        optimizer.apply_gradients(zip(gradients,model.variables))
        
        return policy_loss,value_loss,entropy_loss
