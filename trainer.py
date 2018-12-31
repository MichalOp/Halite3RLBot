from random import choice, shuffle,random
import numpy as np
from time import time,sleep
from multiprocessing import Queue, Pipe, Process, Value
import queue
from envcontrol import *
from threading import Thread

batchsize = 128

def generate(model, queues, trajectories, workers_list):
    
    task_queue,trajectory_queue,reward_queue,kills_matter, total_episodes = queues
    
    while True:
        
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
        
        sleep(1.0)
    
    
def train(model, trajectories, reward_queue, train_func, tf, kills_matter):
    
    old_trajectories = [[],[],[]]
    
    global_step = tf.train.get_or_create_global_step()
    
    summary_writer = tf.contrib.summary.create_file_writer(
        "training", flush_millis=10000)
    
    while True:
        
        while len(trajectories) < 12500:
            sleep(1.0)
        
        if global_step>0:
            kills_matter.value = 1
        
        all_trajectories = old_trajectories[0]+trajectories +old_trajectories[1]+old_trajectories[2]
        old_t = time()
        summary = tf.Summary()
        
        
        old_trajectories[2] = old_trajectories[1]
        old_trajectories[1] = old_trajectories[0]
        old_trajectories[0] = list(trajectories)
        
        trajectories.clear()
        
        shuffle(all_trajectories)
        
        shapes = {32:[],48:[],64:[]}
        
        for t in all_trajectories:
            shapes[t[0].shape[0]].append(t)
        
        print(len(all_trajectories))
        
        prep = 0
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
            if not len(batch) == batchsize:
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
            
            policy_loss, value_loss, entropy_loss = train_func(model,boards,actions,probabilities,advantages,values,masks)
            
            policy_loss_counter+= float(policy_loss)
            value_loss_counter+= float(value_loss)
            entropy_loss_counter+= float(entropy_loss)
            
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
            
        model.save_weights('./weights/halite_model')
        global_step.assign_add(1)
        print("-------------------------------------------------------------------")
        print(global_step)
        print("-------------------------------------------------------------------")
        #print("time: "+str(time()-old_t))
        print("preparation: "+str(prep))
        print("optimizing: "+str(opt))
    

def run_training():
    
    np.set_printoptions(threshold=np.inf)
    
    
    reward_queue = Queue()
    
    trajectory_queue = Queue()
    task_queue = Queue()
    kills_matter = Value('i')
    kills_matter.value = 0
    
    total_episodes = Value('i')
    total_episodes.value = 0
    
    queues = task_queue,trajectory_queue,reward_queue,kills_matter, total_episodes
    
    workers_list = create_workers(queues)
    
    import tensorflow as tf
    from halite_network import Model
    from train_func import train_func
    
    tf.enable_eager_execution()
    
    model = Model()
    #model.load_weights('./weights/halite_model')
    
    trajectories = []
    
    g = lambda:generate(model,queues, trajectories, workers_list)
    t = lambda:train(model, trajectories, reward_queue, train_func,tf, kills_matter)
    
    t1 = Thread(target = g)
    t2 = Thread(target = t)
    
    t1.start()
    t2.start()
        
            #steps+=1
if __name__ == "__main__":

    run_training()
#trajectories, values = generator.generate_trajectories(2000)


#for t in trajectories:
#    for x in t[1:]:
#        print(x)



#print(trajectories)

#print(values)
            
            
            
            
            
            
            
            
            
    
