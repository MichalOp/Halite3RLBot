import connector
from random import choice, shuffle,random
import numpy as np
import scipy.signal
from time import time,sleep
from multiprocessing import Queue, Pipe, Process, Value
import queue

from cv2 import imshow, waitKey

gamma = 0.995
gae_param = 0.96

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]



#tracker = SummaryTracker()

class DataHolder():

    def __init__(self,map_size):
        self.map_size = map_size
        self.episode_buffer = []
        self.episode_values = {}
        self.episode_rewards = {}
        self.ships_to_ignore = set()
        self.episode_step_count = 0
        self.episode_reward = 0
        self.episode_max_ship_count = 0
        self.all_ships = {}

    def step(self, board, ships, actions, probability, rewards, values,reward_sum, to_ignore):
        self.ships_to_ignore = self.ships_to_ignore.union(to_ignore)
        for ship in ships:
            if not ship[0] in self.all_ships:
                self.all_ships[ship[0]] = self.episode_step_count
                self.episode_rewards[ship[0]] = []
                self.episode_values[ship[0]] = []
        
        #print("problen "+str(len(probability)))
        
        self.episode_buffer.append([board, {ship[0]:ship[1] for ship in ships},
                                    {ships[i][0]:actions[i] for i in range(len(ships))},
                                    {ships[i][0]:probability[i] for i in range(len(ships))} ])
        
        rewards = {ships[i][0]:rewards[i] for i in range(len(ships))}
        values = {ships[i][0]:values[i] for i in range(len(ships))}
        
        for s in self.all_ships:
            if s in rewards:
                self.episode_values[s].append(values[s])
                self.episode_rewards[s].append(rewards[s])
            else:
                self.episode_rewards[s][-1]+=0 #to be changed
        self.episode_max_ship_count = max(self.episode_max_ship_count,len(ships))
        self.episode_step_count += 1
        self.episode_reward += reward_sum

    def end_episode(self):
        
        M = self.map_size
        
        discounted_rewards = {}
        advantages = {}
        
        for ship in self.all_ships:
            rewards_plus = np.asarray(self.episode_rewards[ship] + [0.0])
            discounted_rewards[ship] = discount(rewards_plus, gamma)[:-1]
            value_plus = np.asarray(self.episode_values[ship] + [0.0])
            advantages[ship] = self.episode_rewards[ship]+ gamma * value_plus[1:] - value_plus[:-1]
            advantages[ship] = discount(advantages[ship], gamma*gae_param)
        
        trajectories = []
        
        pl = (64-M)//2
        
        for i in range(len(self.episode_buffer)):
            #print(i)
            board = self.episode_buffer[i][0]
            ship_values = np.zeros([M,M,1])
            ship_advantages = np.zeros([M,M])
            ship_actions = np.zeros([M,M,6])
            ship_probabilities = np.zeros([M,M])
            ship_masks = np.zeros([M,M,1])
            ship_positions = []
            
            for ship in self.episode_buffer[i][1]:
                if ship in self.ships_to_ignore:
                    continue
                position = self.episode_buffer[i][1][ship]
                ship_positions.append(position)
                #print(self.all_ships[ship])
                ship_values[position] = [discounted_rewards[ship][i-self.all_ships[ship]]]
                ship_advantages[position] = advantages[ship][i-self.all_ships[ship]]
                ship_actions[position] = self.episode_buffer[i][2][ship]
                ship_probabilities[position] = self.episode_buffer[i][3][ship]
                ship_masks[position] = [1]
            
            board = np.asarray(board,dtype = np.float32)
            #imshow("debug2",ship_masks)
            #waitKey(20)
            #print(ship_masks.shape)
            if len(ship_positions)>0:    
                trajectories.append((board,ship_actions,ship_probabilities,ship_advantages,ship_values,ship_masks))
        
        return trajectories, self.episode_step_count, self.episode_reward,self.episode_max_ship_count

    def reset(self):
        self.ships_to_ignore = set()
        self.episode_buffer = []
        self.episode_values = {}
        self.episode_rewards = {}
        self.episode_step_count = 0
        self.episode_reward = 0
        self.episode_max_ship_count = 0
        self.all_ships = {}


class EnvController():
    
    def __init__(self,trajectory_queue,reward_queue,map_size,players,kills_matter, total_episodes):
        self.total_episodes = total_episodes
        self.kills_matter = kills_matter
        self.reward_queue = reward_queue
        self.trajectory_queue = trajectory_queue
        self.map_size = map_size
        self.players_count = players
        self.data_holders = []
        self.frame_counter = 0
        for p in range(players):
            self.data_holders.append(DataHolder(map_size))
        self.start()
        
    def start(self):
        self.total_episodes.value = self.total_episodes.value+1
        print(self.total_episodes.value)
        save_replay = False
        if self.total_episodes.value%50 == 0:
            save_replay = True
        self.players, self.thread = connector.launch(self.map_size,self.players_count, save_replay)
        self.current_players = list(range(self.players_count))
        self.state = [player.get_game_state() for player in self.players]
        
    def get_state(self):
        out_state = []
        for i in range(self.players_count):
            #print(str(self.state[i][0])+" "+str(len(self.state[i][1])))
            ships, board = self.state[i]
            
            out_ships = []
            for ship_id, position, reward in ships:
                out_ships.append((ship_id,position))
            
            out_state.append((out_ships,board))
        
        return out_state
    
    
    def step(self,actions,probabilities,values):
        self.frame_counter+=1
        for i in range(len(self.current_players)):
            self.players[self.current_players[i]].send_orders(actions[i])
        
        hopeful_positions = []
        dropped = set()
        for posgroup, drop in [self.players[p].get_hopeful_positions() for p in self.current_players]:
            if not drop == -1:
                dropped.add(drop)
            if not posgroup == None:
                hopeful_positions+= posgroup
        
        newstate = [self.players[p].get_game_state() for p in self.current_players]
        
        marked_for_removal = []
        
        #print(hopeful_positions)
        
        possible_kills = {}
        for s, pos in hopeful_positions:
            if pos in possible_kills:
                possible_kills[pos].append(s)
            else:
                possible_kills[pos] = [s]
        
        ship_values = {}
        
        for i in range(len(self.current_players)):
            ships, _ = self.state[i]
            for s, v in zip(ships,values[i]):
                ship_values[s[0]] = v
        
        #print(ship_values)
        
        kills = {}
        kill_values = {}
        
        for pos in possible_kills:
            if len(possible_kills[pos])==2:
                s1 = possible_kills[pos][0]
                s2 = possible_kills[pos][1]
                kills[s1] = s2
                kills[s2] = s1
                kill_values[s1] = ship_values[s2]
                kill_values[s2] = ship_values[s1]
        
        #for k in kills:
        #    print("killed "+str(k))
        
        for i in range(len(self.current_players)):
            
            p = self.current_players[i]
            
            if newstate[i] == None:
                #print("does that ever happen")
                data_holder = self.data_holders[p]
                t, sc, r,msc = data_holder.end_episode()
                print(len(t))
                for tra in t:
                    self.trajectory_queue.put(tra)
                #self.trajectories +=t
                self.reward_queue.put(r)
                marked_for_removal.append(p)
                data_holder.reset()
            
            else:
            
                ships, board = self.state[i]
                newships,_ = newstate[i]
                newships = {newship[0]:newship[2] for newship in newships}
                my_ships = []
                rewards = []
                act = []
                probs = []
                for x in range(len(ships)):
                    ship,position,_ = ships[x]
                    reward = 0
                    if ship in newships:
                        reward = newships[ship]/10000
                    else:
                        if self.players_count == 2 and self.kills_matter.value > 0 and ship in kill_values:
                            reward = kill_values[ship]
                            print("rewarded "+ str(reward)+" for kill by" + str(ship))
                    #probs.append(probabilities[i][x])
                    my_ships.append((ship,position))
                    rewards.append(reward)
                    act.append(actions[i][x][1])
                
                
                self.data_holders[p].step(np.asarray(board),my_ships,act,probabilities[i], rewards,values[i], sum(rewards)*10000, dropped)
            
        if (not len(marked_for_removal) == 0) and (not len(marked_for_removal) == self.players_count):
            with open("debugpleasework",'a') as f:
                f.write("here we go\n")
                import datetime
                f.write(str(datetime.datetime.now())+'\n')
        
        for x in marked_for_removal:
            i = self.current_players.index(x)
            del newstate[i]
            self.current_players.remove(x)
        
        if len(self.current_players) == 0:
            #print("ended, restarting")
            self.start()
        else:
            self.state = newstate


    
    
                
class TrajectoryGenerator:

    def __init__(self, my_id, inpipe, players_count, map_size, global_params):
        task_queue, trajectory_queue, reward_queue, kills_matter, total_episodes = global_params
        self.players = players_count
        self.map_size = map_size
        self.generator_id = my_id
        self.pipe = inpipe
        self.task_queue = task_queue
        self.env_controller = EnvController(trajectory_queue,reward_queue,map_size,players_count,kills_matter,total_episodes)
    
    def run_step(self):
        
        boards, ship_ids, positions = [],[],[]
        

        state = self.env_controller.get_state()
        current_players = 0
        for ships, board in state:
            current_players+=1
            #print(ships)
            s, pos = [],[]
            for ship,position in ships:
                #print(ship)
                s.append(ship)
                pos.append(position)
            
            boards.append(board)
            ship_ids.append(s)
            positions.append(pos)
        
        
        #print(np.asarray(boards).shape)
        padded = np.asarray(boards,dtype=np.float32)
        
        assert not current_players == 0, "that should definitely not happen"
        
        self.task_queue.put((self.generator_id,current_players,padded))
        
        all_probs, all_values = self.pipe.recv()
        #print(len(all_probs))
        #imshow("debug",all_values[0]/np.max(all_values[0]))
        #waitKey(1)
        
        actions,probabilities,state_values = [],[],[]
        
        for i in range(current_players):
            
            probs = all_probs[i]
            values = all_values[i]
            #probs = probs.numpy()
            #values = values.numpy()
            #print(values)
            #print(probs)
            action = []
            result_probs = []
            pos = positions[i]
            vals = []
            for position in pos:
                pr = probs[position]
                value = values[position]
                
                #pr = np.asarray(pr)
                pr = pr/np.sum(pr)
                if random()<0.98:
                    t = np.random.choice((0,1,2,3,4,5),p=pr)
                else:
                    t = np.random.choice((0,1,2,3,4,5))
                act = [0]*6
                act[t] = 1
                action.append(act)
                result_probs.append(pr[t])
                vals.append(value)
            
            for x in range(len(ship_ids[i])):
                action[x] = (ship_ids[i][x],action[x])
            
            #print(action)
            #print(result_probs)
            actions.append(action)
            probabilities.append(result_probs)
            state_values.append(np.asarray(vals))
            
        self.env_controller.step(actions,probabilities,state_values)

            
    
    def generate_trajectories(self):
        
        while True:
            self.run_step()
                
batchsize = 256

def launch_worker(pipe,process_id,players,size,global_params):
    generator = TrajectoryGenerator(process_id,pipe,players,size,global_params)
    generator.generate_trajectories()
        
def create_worker(workers_list,map_size,players, queues):
    a, b = Pipe()
    process = Process(target = launch_worker,args = (b,len(workers_list),players,map_size, queues))
    process.start()
    workers_list.append(a)
        
def create_workers(queues):
    workers_list = []
    
    create_worker(workers_list,32,2,queues)
    create_worker(workers_list,32,2,queues)
    #create_worker(workers_list,32,2,queues)
    #create_worker(workers_list,32,2,queues)
    create_worker(workers_list,48,2,queues)
    #create_worker(workers_list,48,2,queues)
    create_worker(workers_list,48,2,queues)
    #create_worker(workers_list,32,4,queues)
    create_worker(workers_list,32,4,queues)
    create_worker(workers_list,48,4,queues)
    
    return workers_list

def compute_task_batch(model, task_queue, workers_list):
    #print(started)
    tasks = []
    ids = []
    player_counts = []
    
    while True:
        val = None
        try:
            val = task_queue.get(timeout = 0.001)
        except queue.Empty:
            break
        index,players, task = val
        ids.append(index)
        player_counts.append(players)
        tasks.append(task)
    
    
    if len(tasks)>0:
        
        policy = []
        value = []
        
        for t in tasks:
            p, v = model(t)
            policy.append(p.numpy())
            value.append(v.numpy())
        
        #tasks = np.concatenate(tasks,0)
        #print(tasks.shape)
        #print(model)
        #input()
        
        iterator = 0
        
        for i in range(len(ids)):
            pc = player_counts[i]
            workers_list[ids[i]].send((policy[i],value[i]))
            iterator+=player_counts[i]

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
            
            
            
            
            
            
            
            
            
    
