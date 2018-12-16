#!/usr/bin/env python3
# Python 3.6
import os, sys
# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction,Position

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging
from time import sleep,time
from math import ceil
from queue import PriorityQueue, Queue
import numpy as np


pipe_id = sys.argv[1]

#print(pipe_id)
#print(pipe_id)
#print(pipe_id)
#print(pipe_id)
#exit()

pipe_in = open("/tmp/halite_commands"+pipe_id, 'r')
pipe_out = open("/tmp/halite_data"+pipe_id, 'w')

move_array = [(0,0),(1,0),(0,1),(-1,0),(0,-1)]

game = hlt.Game()

bx = game.game_map.width
by = game.game_map.height

def to_tuple(p):
    return (p.x%bx,p.y%by)

dropoffs = set()

dropoffs.add(to_tuple(game.me.shipyard.position))


# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("MyPythonBot")



ships_last_cargo = {}



try_spawn = False

def movement_solver(ships_and_targets, game_map, block_fields, the_end = False, bases = None):
    board = [[[-1,0,0,False,[]] for x in range(game_map.width)]for y in range(game_map.height)]
    
    ship_dict = {}
    
    ship_list = []
    
    to_block = Queue()
    
    for x,y in block_fields:
        board[x][y][3] = True
    
    to_consider = set()
    
    for ship, target_preference in ships_and_targets:
        x,y= to_tuple(ship.position)
        ship_dict[ship.id] = ship
        #logging.info(str(target_preference))
        if ship.halite_amount < ceil(game_map[ship.position].halite_amount*0.1):
            board[x][y][0] = ship.id
            board[x][y][1] = 0
            board[x][y][3] = True
            
        else:
            board[x][y][0] = ship.id
            target = np.argmax(target_preference)
            #logging.info(f'target: {target}')
            board[x][y][1] = target
            if target == 0:
                board[x][y][3] = True
            else:
                to_consider.add((x,y))
                mx,my = move_array[target]
                board[(x+mx)%bx][(y+my)%by][4].append((x,y))
        
    i = 0
    
    if the_end:
        for x,y in bases:
            board[x][y][3]=False
            board[x][y][0] = -1
    
    endpoints = set()
    
    while to_consider:
        i+=1
        x,y = next(iter(to_consider))
        
        potential_block = []
        
        while not (board[x][y][0]== -1 or not board[x][y][2] == 0 or board[x][y][3]):
            to_consider.remove((x,y))
            potential_block.append((x,y))
            board[x][y][2] = i
            mx,my = move_array[board[x][y][1]]
            x, y = ((x+mx)%bx,(y+my)%by)
        
        if board[x][y][3]:
            for px, py in potential_block:
                board[px][py][3] = True
                board[px][py][1] = 0
            continue
        
        if board[x][y][0]==-1:
            #logging.info(f'added endpoint {x} {y}')
            endpoints.add((x,y))
            continue
        
        if board[x][y][2] == i:
            first_on_cycle = potential_block.index((x,y))
            if not first_on_cycle == 0:
                for px, py in potential_block[:first_on_cycle]:
                    board[px][py][3] = True
                    board[px][py][1] = 0
            
            for px, py in potential_block:
                board[px][py][3] = True
    
    def helper(x,y):
        #logging.info(f'running helper for endpoint {x} {y}')
        strings = [helper(px,py) for px,py in board[x][y][4]]
        max_len = 0
        i = 0
        index = 0
        winning_string = []
        if the_end and (x,y) in bases:
            for s in strings:
                winning_string+=s
            #logging.info(f'returning string {winning_string}')
        else:
            for s in strings:
                if len(s)>max_len:
                    max_len = len(s)
                    index = i
                    winning_string = s
                    
                i+=1
        
            if len(strings)>0:
                del strings[index]
            
            for s in strings:
                for u, v in s:
                    board[u][v][3] = True
                    board[u][v][1] = 0
            
        winning_string.append((x,y))
        
        return winning_string
    
    while endpoints:
        x,y = next(iter(endpoints))
        
        endpoints.remove((x,y))
        
        winning_string = helper(x,y)[:-1]
        
        for px, py in winning_string:
            board[px][py][3] = True
        
    orders_list = []
    
    hopeful_positions = []
    
    for x in range(bx):
        for y in range(by):
            ship_id = board[x][y][0]
            move = board[x][y][1]
            
            if not ship_id == -1:
                orders_list.append(ship_dict[ship_id].move(move_array[move]))
                position = to_tuple(ship_dict[ship_id].position.directional_offset(move_array[move]))
                if not the_end:
                    hopeful_positions.append((ship_id,position))
                if not board[x][y][3]:
                    logging.info(f"what {x} {y}")
    
    return orders_list, hopeful_positions


def desired_return_pathing(map,dropoffs):
    
    global bx
    global by
    
    map_representation = [[None for x in range(by)]for y in range(bx)]
    
    logging.info(len(map_representation))
    
    q = PriorityQueue()
    visited = set()
    
    for d in dropoffs:
        q.put((0,0,d,0))
    
    

    counter = 0

    #logging.info("assigned targets:")

    #old_t = time()

    while not q.empty():
        counter+=1
        cost,distance,pos,move_direction = q.get()
        if pos in visited:
            continue
        
        visited.add(pos)
        
        position = Position(pos[0],pos[1])
        #logging.info(position)
        map_representation[position.x%bx][position.y%by] = (cost-map[position].halite_amount*0.1,
                                                        distance,
                                                        map[position].halite_amount,
                                                        move_direction)
        
        
        move_cost = ceil(map[position].halite_amount*0.1)
        for i in range(4):
            x = position.directional_offset(move_array[i+1])
            if (x.x%bx,x.y%by) not in visited:
                q.put((cost+map[x].halite_amount*0.2+3.0,distance+1,(x.x%bx,x.y%by),(i+2)%4+1))

    logging.info("iterations {}".format(str(counter)))
    
    return map_representation

return_pathing = None
refresh_return = False

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map
    
    shipdict = {}
    
    game_end = False
    
    if max_turns[bx]-game.turn_number<=bx/2:
        game_end = True
    
    if game.turn_number%20 == 1 or refresh_return:
        refresh_return = False
        return_pathing = desired_return_pathing(game_map,dropoffs)
    
    pipe_out.write(str(game.turn_number)+" "+str(me.halite_amount)+"\n")
    
    for ship in me.get_ships():
        
        reward = 0
        shipdict[ship.id] = ship
        
        if ship.id not in ships_last_cargo:
            ships_last_cargo[ship.id] = 0
        
        if to_tuple(ship.position) in dropoffs:
            reward = ships_last_cargo[ship.id]
        
        ships_last_cargo[ship.id] = ship.halite_amount

        pipe_out.write("{} {} {} {} ".format(ship.id,ship.position.x%bx, ship.position.y%by, reward))
    
    pipe_out.write("\n")
    #pipe_out.flush()
    
    for i in range(bx):
        for j in range(by):
            
            cell = game_map[Position(i,j)]
            
            s = ""
            
            if (i,j) in dropoffs:
                s+="1000 "
            else:
                s+="0 "
            
            if cell.is_occupied:
                ship = cell.ship
                if ship.owner == me.id:
                    s += "1000 0 "
                else:
                    s += "0 1000 "
                
                s += str(ship.halite_amount)+" "
            else:
                s += "0 0 0 "
                
            s += str(cell.halite_amount) + " "+str(return_pathing[i][j][0])
            
            
            pipe_out.write(s+"\n")
            #pipe_out.flush()
    
    pipe_out.flush()
    
    ready_commands = []
    
    orders_list = []
    
    block_fields = []
    
    real_halite_amount = me.halite_amount
    #while len(in_data)== 0:
    #    in_data = pipe_in.readline().strip()
    
    selected_ship = None
    #print(len(dropoffs))
    if len(me.get_ships()) < len(dropoffs)*15:
        if game.turn_number < 250:
            try_spawn = True
    else:
        selected_ship = random.choice(list(me.get_ships()))
        
    ship_dropped = -1
    
    for i in range(len(me.get_ships())):
        data = pipe_in.readline().strip().split()
        
        ship = shipdict[int(data[0])]
        returning = int(data[1])
        values = [float(x) for x in data[2:]]
        
        if selected_ship and ship.id == selected_ship.id and not game_map[ship.position].has_structure and real_halite_amount >= constants.DROPOFF_COST:
            ship_dropped = ship.id
            ready_commands.append(ship.make_dropoff())
            dropoffs.add(to_tuple(ship.position))
            real_halite_amount -= 4000
            refresh_return = True
            continue
        
        if returning == 1:
            
            _,_,_,direction = return_pathing[ship.position.x%bx][ship.position.y%by]
            move_value = [0]*5
            
            move_value[direction] = 1
            #logging.info("ship "+str(ship.id)+" returning in dir "+str(np.argmax(move_value)))
            orders_list.append((ship,move_value))
        else:
            orders_list.append((ship,values))
    
    spawning_ship = False
    
    if try_spawn and real_halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        try_spawn = False
        spawning_ship = True
    
    if spawning_ship:
        block_fields.append((me.shipyard.position.x,me.shipyard.position.y))
    
    command_queue, hopeful_positions = movement_solver(orders_list,game_map,block_fields,game_end,dropoffs)
    
    pipe_out.write(str(ship_dropped))
    
    pipe_out.write("\n")
    
    for s, p in hopeful_positions:
        pipe_out.write("{} {} {} ".format(s,p[0], p[1]))
    
    pipe_out.write("\n")    
    
    if spawning_ship:
        command_queue.append(game.me.shipyard.spawn())
    
    command_queue += ready_commands
    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
