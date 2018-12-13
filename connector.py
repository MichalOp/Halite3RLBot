import os, sys, subprocess
import random
import numpy as np
import threading

import signal
import time

from cv2 import imshow, waitKey

max_turns = {32:400,40:425,48:450,56:475,64:500}

class player:
    def __init__(self,pipe_id,map_size,process):
        self.pipe_id = pipe_id
        self.max_turn = 500
        self.process = process
        self.map_size = map_size
        self.money = 5000
        self.money_delta = 0
        #print("launching")
        self.pipe_out = open("/tmp/halite_commands"+pipe_id, 'w')
        self.pipe_in = open("/tmp/halite_data"+pipe_id, 'r')
    
    def get_game_state(self):
        
        if not self.process.isAlive() or self.pipe_in.closed:
            self.clear()
            return None        
        
        #while True:
        #    print(self.pipe_in.read())
        
        
        status = self.pipe_in.readline().strip().split()
        
        if len(status) == 0:
            self.clear()
            return None
        
        #print(status)
        if int(status[0]) == self.max_turn:
            terminal = True
            
        self.money_delta = int(status[1]) - self.money
        self.money = int(status[1])
        
        my_ships = self.pipe_in.readline().strip()
        my_ships = my_ships.split()
        
        ships = []
        
        for i in range(0,len(my_ships),4):
            ships.append((int(my_ships[i]),(int(my_ships[i+1]),int(my_ships[i+2])),int(my_ships[i+3])))
        
        board_data = []
        
        game_progress = float(status[0])/max_turns[self.map_size]*100
        
        #print(game_progress)
        
        for i in range(self.map_size):
            board_row = []
            for j in range(self.map_size):
                cell_info = self.pipe_in.readline().strip().split()
                cell_info.append(game_progress)
                board_row.append([float(x) for x in cell_info])
            board_data.append(board_row)
            
        return ships, board_data
    
    def get_hopeful_positions(self):
        
        if not self.process.isAlive() or self.pipe_in.closed:
            #self.clear()
            return None, -1
        
        ship_dropped = self.pipe_in.readline().strip()
        if len(ship_dropped)>0:
            ship_dropped = int(ship_dropped)
        else:
            ship_dropped = -1
        
        my_ships = self.pipe_in.readline().strip()
        my_ships = my_ships.split()
        
        ships = []
        
        for i in range(0,len(my_ships),3):
            ships.append((int(my_ships[i]),(int(my_ships[i+1]),int(my_ships[i+2]))))
            
        return ships, ship_dropped
        
        
    def send_orders(self, orders_list):
        #self.pipe_out.flush()
        
        for ship,priorities in orders_list:
            #print(ship)
            #print(priorities)
            self.pipe_out.write(str(ship)+" ")
            for x in priorities:
                self.pipe_out.write(str(x)+" ")
            self.pipe_out.write("\n")
        
        self.pipe_out.flush()
        
    def clear(self):
        self.pipe_out.close()
        self.pipe_in.close()
        os.unlink("/tmp/halite_commands"+self.pipe_id)
        os.unlink("/tmp/halite_data"+self.pipe_id)

def launch(map_size, players_count, save_replay = False):
    
    pipe_ids = []
    
    for i in range(players_count):
        pipe_ids.append(str(random.randrange(0,2**31)))

    

    for pipe_id in pipe_ids:
        try:
            os.mkfifo("/tmp/halite_data"+pipe_id)
            os.mkfifo("/tmp/halite_commands"+pipe_id)
        except:
            pass
    
    command_str = "./halite --no-logs --no-timeout --width "+str(map_size)+" --height "+str(map_size)
    if save_replay:
        command_str += " --replay-directory ./run_replays"
    else:
        command_str += " --no-replay"
    for pipe_id in pipe_ids:
        command_str+= " 'python3 PipeBot.py {}'".format(pipe_id)
    
    process = lambda:os.system(command_str)
    
    t = threading.Thread(target = process)
    t.start()
    #process = subprocess.Popen(["./halite","--no-replay","--no-timeout","--width "+str(map_size),"--height "+str(map_size),'"python3 PipeBot.py {}"'.format(pipe_id1),'"python3 PipeBot.py {}"'.format(pipe_id2)])
    
    return [player(pipe_id,map_size,t) for pipe_id in pipe_ids], t

def main():
    while True:
        players, process = launch(64,4)
        
        states = []
        for player in players:
            states.append(player.get_game_state())

        while not states[0] == None:
            
            board = states[0][1]
            board = np.asarray(board)
            #print(np.sum(board[:,:,3]))
            board = board/np.max(board[:,:,3])
            imshow("xd",board[:,:,0:3])
            waitKey(1)
            
            #print(len(x[1]))
            
            for player, state in zip(players,states):
                orders = []
                for s in state[0]:
                    orders.append((s[0],[1,0,0,0,0,0]))
            
                player.send_orders(orders)
            
            states = []
            for player in players:
                states.append(player.get_game_state())

if __name__ == "__main__":
    main()
