import connector
from random import choice, shuffle,random
import numpy as np
import scipy.signal
from time import time,sleep
from multiprocessing import Queue, Pipe, Process, Value
import queue
from looptime import add_t, reset_t, clear_t, log_by_tag

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
        
        reset_t("building trajectories")
        
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
                ship_values[position] = [discounted_rewards[ship][i-self.all_ships[ship]]]
                ship_advantages[position] = advantages[ship][i-self.all_ships[ship]]
                ship_actions[position] = self.episode_buffer[i][2][ship]
                ship_probabilities[position] = self.episode_buffer[i][3][ship]
                ship_masks[position] = [1]
            
            board = np.asarray(board,dtype = np.float32)
            
            if len(ship_positions)>0:    
                trajectories.append((board,ship_actions,ship_probabilities,ship_advantages,ship_values,ship_masks))
        
        add_t("building trajectories")
        
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
        with open("timings"+str(self.map_size)+str(self.players_count),'w') as f:
            log_by_tag(f)
        clear_t()
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
            ships, board = self.state[i]
            
            out_ships = []
            for ship_id, position in ships:
                out_ships.append((ship_id,position))
            
            out_state.append((out_ships,board))
        
        return out_state
    
    
    def step(self,actions,probabilities,values):
        self.frame_counter+=1
        reset_t("sending orders")
        for i in range(len(self.current_players)):
            self.players[self.current_players[i]].send_orders(actions[i])
        add_t("sending orders")
        
        reset_t("hopeful positions")
        hopeful_positions = []
        dropped = set()
        ship_parent = {}
        
        for p in self.current_players:
            posgroup, drop = self.players[p].get_hopeful_positions()
            if not drop == -1:
                dropped.add(drop)
            if not posgroup == None:
                for s,_,_ in posgroup:
                    ship_parent[s] = p
                hopeful_positions+= posgroup
        
        
        
        add_t("hopeful positions")
        reset_t("getting game state")
        newstate = [self.players[p].get_game_state() for p in self.current_players]
        add_t("getting game state")
        marked_for_removal = []
        
        #print(hopeful_positions)
        reset_t("all this autism")
        possible_kills = {}
        ship_rewards = {}
        for s, pos,r in hopeful_positions:
            ship_rewards[s] = r
            if pos in possible_kills:
                possible_kills[pos].append(s)
            else:
                possible_kills[pos] = [s]
        
        ship_values = {}
        
        for i in range(len(self.current_players)):
            ships, _ = self.state[i]
            for s, v in zip(ships,values[i]):
                ship_values[s[0]] = v
        
        kills = {}
        kill_values = {}
        
        for pos in possible_kills:
            if len(possible_kills[pos])==2:
                s1 = possible_kills[pos][0]
                s2 = possible_kills[pos][1]
                if not ship_parent[s1] == ship_parent[s2]:
                    kills[s1] = s2
                    kills[s2] = s1
                    kill_values[s1] = ship_values[s2]
                    kill_values[s2] = ship_values[s1]
        
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
                newships = {newship[0]:newship[1] for newship in newships}
                my_ships = []
                rewards = []
                act = []
                probs = []
                for x in range(len(ships)):
                    ship,position = ships[x]
                    reward = 0
                    if ship in ship_rewards:
                        reward = ship_rewards[ship]/10000
                    if not ship in newships:
                        if self.players_count == 2 and self.kills_matter.value > 0 and ship in kill_values:
                            reward = kill_values[ship]
                            print("rewarded "+ str(reward)+" for kill by" + str(ship))

                    my_ships.append((ship,position))
                    rewards.append(reward)
                    act.append(actions[i][x][1])
                
                
                self.data_holders[p].step(np.asarray(board),my_ships,act,probabilities[i], rewards,values[i], sum(rewards)*10000, dropped)
        
        for x in marked_for_removal:
            i = self.current_players.index(x)
            del newstate[i]
            self.current_players.remove(x)
        
        if len(self.current_players) == 0:
            self.start()
        else:
            self.state = newstate

        add_t("all this autism")
    
    
                
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
        
        reset_t("await network")
        
        #print(np.asarray(boards).shape)
        padded = np.asarray(boards,dtype=np.float32)
        
        assert not current_players == 0, "that should definitely not happen"
        
        self.task_queue.put((self.generator_id,current_players,padded))
        
        all_probs, all_values = self.pipe.recv()
        
        add_t("await network")
        
        actions,probabilities,state_values = [],[],[]
        
        reset_t("retrieve actions")
        
        for i in range(current_players):
            
            
            
            probs = all_probs[i]
            values = all_values[i]
            
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
            
            actions.append(action)
            probabilities.append(result_probs)
            state_values.append(np.asarray(vals))
        
        add_t("retrieve actions")
        
        self.env_controller.step(actions,probabilities,state_values)
    
    def generate_trajectories(self):
        
        while True:
            reset_t("total")
            self.run_step()
            add_t("total")

def compute_task_batch(model, task_queue, workers_list):
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
        
        iterator = 0
        
        for i in range(len(ids)):
            pc = player_counts[i]
            workers_list[ids[i]].send((policy[i],value[i]))
            iterator+=player_counts[i]

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
    create_worker(workers_list,64,2,queues)
    create_worker(workers_list,48,2,queues)
    create_worker(workers_list,64,4,queues)
    create_worker(workers_list,32,4,queues)
    create_worker(workers_list,48,4,queues)
    
    return workers_list            
            
            
            
            
            
            
            
            
    
