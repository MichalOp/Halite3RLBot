from random import choice, shuffle,random
import numpy as np
from time import time,sleep
from multiprocessing import Queue, Pipe, Process, Value
from envcontrol import *
import queue

batchsize = 256

def run_training():
    
    np.set_printoptions(threshold=np.inf)
    
    trajectory_queue = Queue()
    task_queue = Queue()
    reward_queue = Queue()
    
    kills_matter = Value('i')
    kills_matter.value = 0
    
    total_episodes = Value('i')
    total_episodes.value = 0
    
    queues = task_queue,trajectory_queue,reward_queue,kills_matter, total_episodes
    
    workers_list = create_workers(queues)
    
    import tensorflow as tf
    from halite_network import Model
    
    tf.enable_eager_execution()
    
    global_step = tf.train.get_or_create_global_step()
    summary_writer = tf.contrib.summary.create_file_writer(
        "training", flush_millis=10000)
    
    optimizer = tf.train.AdamOptimizer(1e-4)
    
    model = Model()
    #model.load_weights('./weights/halite_model_v9')
    old_trajectories = [[],[],[]]
    
    def normalize_loss(loss):
        return loss / (tf.abs(tf.stop_gradient(loss)) + 0.5)
    
    while True:
        
        if global_step>200:
            kills_matter.value = 1
        
        trajectories = []
        
        while len(trajectories)<12500:
            #print(len(trajectories))
            compute_task_batch(model,task_queue,workers_list)
            
            while True:
                trajectory = None
                #print(trajectory_queue.qsize())
                try:
                    trajectory = trajectory_queue.get(timeout = 0.001)
                except queue.Empty:
                    break
                trajectories.append(trajectory)
                
            sleep(0.0001)
        
        all_trajectories = old_trajectories[0]+trajectories +old_trajectories[1]+old_trajectories[2]
        old_t = time()
        summary = tf.Summary()
        
        
        old_trajectories[2] = old_trajectories[1]
        old_trajectories[1] = old_trajectories[0]
        old_trajectories[0] = trajectories
        
        shuffle(all_trajectories)
        
        shapes = {32:[],48:[],64:[]}
        
        for t in all_trajectories:
            shapes[t[0].shape[0]].append(t)
        
        print(len(all_trajectories))
        
        prep = 0
        run = 0
        applying_mask = 0
        policy_t = 0
        value_t = 0
        entropy_t = 0
        losses= 0
        opt = 0
        
        old_t = time()
        
        policy_loss_counter = 0
        value_loss_counter = 0
        entropy_loss_counter = 0
        
        cut_batches = []
        
        for i in shapes:
            all_trajectories = shapes[i]
            for index in range(len(all_trajectories)//batchsize+1):
                batch = all_trajectories[index*batchsize:min((index+1)*batchsize,len(all_trajectories))]
                cut_batches.append(batch)
        
        shuffle(cut_batches)
                
        for batch in cut_batches:
            if len(batch) == 0:
                continue
            #(board,ship_positions,ship_actions,ship_probabilities,ship_advantages,ship_values)
            boards = []
            actions = []
            probabilities = []
            advantages = []
            values = []
            masks = []
            
            for t in batch:
                board,act,prob,adv,val,mask = t
                boards.append(board)
                actions.append(act)
                probabilities.append(prob)
                advantages.append(adv)
                values.append(val)
                masks.append(mask)
            
            #boards = np.asarray(
            #ship_positions = ship_positions)
            boards = np.asarray(boards,dtype = np.float32)
            actions = np.asarray(actions,dtype=np.float32)
            probabilities = np.asarray(probabilities,dtype=np.float32)
            advantages = np.asarray(advantages,dtype=np.float32)
            values = np.asarray(values,dtype=np.float32)
            masks = np.asarray(masks,dtype=np.float32)
            
            new_t = time()
            prep+= new_t-old_t
            old_t = new_t
            
            #print(values.device)
            
            
            with tf.GradientTape() as tape:
            
                policy, value = model(boards)
                
                
                new_t = time()
                run+= new_t-old_t
                old_t = new_t
                
                total_loss = 0
                

                value = value*masks
                
                new_t = time()
                applying_mask+= new_t-old_t
                old_t = new_t
                
                #policy = policy*masks
                value_loss = tf.nn.l2_loss(value-values)
                #print(value_loss)
                value_loss_counter+= float(value_loss)
                
                new_t = time()
                value_t+= new_t-old_t
                old_t = new_t
                
                responsible_outputs = tf.reduce_sum(actions*policy,3)
                
                ratios = (responsible_outputs + 1e-8)/(probabilities + 1e-8)
                #print(ratios)
                policy_loss = -tf.reduce_sum(tf.minimum(
                                            tf.clip_by_value(ratios, 1/200, 200) * advantages,
                                            tf.clip_by_value(ratios, 1.0 - 0.1, 1.0 + 0.1)*advantages))
                #print(policy_loss)
                print(f"ratios {np.max(ratios.numpy())}")
                
                if abs(float(policy_loss))>10000 and global_step>100:
                    with open("what the hell", 'w') as f:
                        #f.write(f'HALP {boards}\n\n\n')
                        f.write(f'BOARDS\n {boards}\n\n\n')
                        f.write(f'PROBABILITIES\n {probabilities}\n\n\n')
                        f.write(f'RATIOS\n {ratios}\n\n\n')
                        f.write(f'ADVANTAGES\n {advantages}\n\n\n')
                        f.write(f'max ratios: {np.max(ratios.numpy())} min probabilities: {np.min(probabilities)}\n\n\n')
                        
                    input()
                    input()    
                
                policy_loss_counter+= float(policy_loss)
                
                new_t = time()
                policy_t+= new_t-old_t
                old_t = new_t
                
                entropy_loss = tf.reduce_sum(policy * tf.log(policy + 1e-12))
                
                entropy_loss_counter+= float(entropy_loss)
                
                new_t = time()
                entropy_t+= new_t-old_t
                old_t = new_t
                
                #print(entropy_loss)
                    
                #total_loss = 0.5*normalize_loss(value_loss)+normalize_loss(policy_loss)+normalize_loss(entropy_loss)*0.01
                
                regularization = tf.reduce_sum([tf.reduce_sum(tf.abs(x)) for x in model.variables])
                
                total_loss = value_loss+policy_loss+0.0002*regularization + 0.000001*entropy_loss
                
                print(float(total_loss))
                
                new_t = time()
                losses+= new_t-old_t
                old_t = new_t
                
                gradients,norm = tf.clip_by_global_norm(tape.gradient(total_loss,model.variables),2000)
                #print(norm)
                optimizer.apply_gradients(zip(gradients,model.variables))
                
                new_t = time()
                opt+= new_t-old_t
                old_t = new_t
        
        all_trajectories = []
        rewards = []

        while True:
            reward = None
            #print(trajectory_queue.qsize())
            try:
                reward = reward_queue.get(timeout = 0.001)
            except queue.Empty:
                break
            
            rewards.append(reward)
        
        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('Losses/Var_Norm',tf.reduce_sum([tf.reduce_sum(tf.abs(x)) for x in model.variables]))
            tf.contrib.summary.scalar('Losses/Value Loss', value_loss_counter)
            tf.contrib.summary.scalar('Losses/Policy Loss', policy_loss_counter)
            tf.contrib.summary.scalar('Losses/Entropy', entropy_loss_counter)
            tf.contrib.summary.scalar('Performance/AvgReward', np.mean(rewards))
            
        model.save_weights('./weights_3/halite_model_v9')
        global_step.assign_add(1)
        print("time: "+str(time()-old_t))
        print("preparation: "+str(prep))
        print("running: "+str(run))
        print("applying damned mask: "+str(applying_mask))
        print("value loss: "+str(value_t))
        print("policy loss: "+str(policy_t))
        print("entropy: "+str(entropy_t))
        print("optimizing: "+str(opt))
        
            #steps+=1
if __name__ == "__main__":

    run_training()
#trajectories, values = generator.generate_trajectories(2000)


#for t in trajectories:
#    for x in t[1:]:
#        print(x)



#print(trajectories)

#print(values)
            
            
            
            
            
            
            
            
            
    
