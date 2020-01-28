#!/usr/bin/env python3
import logging
logging.getLogger('tensorflow').disabled = True
import fire
import json
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0
import random
import pickle
from pyinflect import getInflection
import textwrap

import model, sample, encoder

def wrap_print(body='', wrap_length=80 ):
    if isinstance(body, str): 
        body = '\n'.join(['\n'.join(textwrap.wrap(line, wrap_length, break_long_words=False, replace_whitespace=False)) for line in body.splitlines() if line.strip() != ''])
        print(body)
    else:
        print(body)

def create_graph(
        nodes=10,
        edges=20,
):
    neighbor_sets = [ set() for i in range(0,nodes) ]
    for i in range(0,nodes-1):
        end_node=i+1            
        start_node=i
        neighbor_sets[start_node].add(end_node)
        neighbor_sets[end_node].add(start_node)
        
    for i in range(nodes+1,edges+1):
        start_node=random.randint(0,nodes-1)
        end_node=start_node
        while end_node==start_node:
            end_node=min(max(start_node + random.randint(-3,3),0),nodes-1)
        neighbor_sets[start_node].add(end_node)
        neighbor_sets[end_node].add(start_node)
    return neighbor_sets

def description_cleanup(
        text=''
):
    # remove the final sentence fragment
    text=".".join(text.split(".")[:-1])
    text = text + '.'
    # only keep what is before <|end of text|>
    text2=text.split("<|endoftext|>")
    text3=text2[0] 
    #get rid of lines that don't end in a period (often a subtitle of some kind)
    paragraphs=text3.split('\n')
    titles_removed = ""
    for paragraph in paragraphs:
        if paragraph.endswith("."):
            #get rid of leading and trailing quote marks, and leading numbers, periods, dashes, parentheses or spaces.
            # paragraph = paragraph.strip('"')
            paragraph = paragraph.lstrip('123456789.- )(')            
            titles_removed = titles_removed + paragraph + "\n"
    return titles_removed

def other_cleanup(
        text=''
):
    # remove the final sentence fragment
    text=".".join(text.split(".")[:-1])
    text = text + '.'
    # only keep what is before <|end of text|>
    text2=text.split("<|endoftext|>")
    text3=text2[0] 
    #get rid of lines that don't end in a period (often a subtitle of some kind)
    paragraphs=text3.split('\n')
    titles_removed = ""
    for paragraph in paragraphs:
        if paragraph.endswith("."):           
            titles_removed = titles_removed + paragraph + "\n"
    return titles_removed

def rooms_cleanup(
        text=''
):
    # keep only up to the first hard return
    rooms_line_split=text.split("\n")
    line_one=rooms_line_split[0]
    # keep only what's before <|end of text|>
    text2=line_one.split("<|endoftext|>")
    line_one=text2[0]
    #separate each room into its own string
    rooms_split=line_one.split(", ")
    #get rid of leading and trailing spaces
    rooms2 = list(map(str.strip, rooms_split))
    
    fixed_array=[]
    for test_string in rooms2:
        if (len(test_string) < 30 and test_string.count(' ')<3):
            fixed_array.append(test_string)
    rooms3=list(fixed_array)
    rooms3 = list(filter(None, rooms3))
    return rooms3

class DescriptionGen():

    def __init__(self, sess, length=115, temperature=0.7, top_k=30):
    
        seed = None
        batch_size=1
        model_path='1558M'
        self.sess = sess
    
        self.enc = encoder.get_encoder(model_path)
        hparams = model.default_hparams()
        with open(os.path.join('models/1558M', 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))  

        self.context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=self.context,
            batch_size=batch_size,
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint('models/1558M')
        saver.restore(self.sess, ckpt)
            
        
    def generate(self, prompt):
        context_tokens = self.enc.encode(prompt)
        if len(context_tokens)>(1023-175):
            context_tokens = context_tokens[-(1023-175):]
        out = self.sess.run(self.output, feed_dict={
                self.context: [context_tokens for _ in range(1)]
            })[:, len(context_tokens):]

        text = self.enc.decode(out[0])
        return text


class GetGen():

    def __init__(self, sess, length=10, temperature=0.7, top_k=1):
    
        seed = None
        batch_size=1
        model_path='1558M'
        self.sess = sess
    
        self.enc = encoder.get_encoder(model_path)
        hparams = model.default_hparams()
        with open(os.path.join('models/1558M', 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))  

        self.context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=self.context,
            batch_size=batch_size,
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint('models/1558M')
        saver.restore(self.sess, ckpt)
            
        
    def generate(self, prompt):
        context_tokens = self.enc.encode(prompt)
        out = self.sess.run(self.output, feed_dict={
                self.context: [context_tokens for _ in range(1)]
            })[:, len(context_tokens):]

        text = self.enc.decode(out[0])
        return text

def interact_model(
    model_name='1558M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=150,
    temperature=1,
    top_k=0,
):

    os.environ['KMP_WARNINGS'] = 'off'
    inventory = set()
    if batch_size is None:
        batch_size = 1
        assert nsamples % batch_size == 0
    
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    
    f=open("src/carrying.txt", "r", errors='ignore')
    if f.mode == 'r':
        carrying_prompt =f.read()
    f=open("src/items.txt", "r", errors='ignore')
    if f.mode == 'r':
        items_prompt =f.read()
    f=open("src/animate.txt", "r", errors='ignore')
    if f.mode == 'r':
        animate_prompt =f.read()
    
    config = tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count={'CPU': 32})
#   with tf.Session(config=config, graph=tf.Graph()) as sess:
    with tf.Session(graph=tf.Graph()) as sess:
        print("The next two loads take about 10 minutes. Sorry for the wait.\nLoading description generator...")
        description_gen = DescriptionGen(sess)
        print("Loading get generator...")
        get_gen = GetGen(sess)

        
        #context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        print("***")
        #saver = tf.train.Saver()
        #ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        #saver.restore(sess, ckpt)
        msg = (
            "\n\n\n\n\n\n\n\n\n\n\nINFINITE ADVENTURE\n\n\n\n"
            "INSTRUCTIONS: Infinite Adventure is primarily an exploration "
            "game. You can type anything you want at the prompt, as "
            "long as it starts with a verb. A few verbs have special effects:\n\n"
            " * go LOCATION -- takes you to that place. If the place already has a "
            "description, it appears down below. If not, you can still go "
            "there if it appears in the description but it takes a minute "
            "to generate the new description.\n\n"
            " * get OBJECT -- allows you to pick up some objects that appear in the "
            "description. If they are too big or can't be picked up, the "
            "system will tell you.\n\n"
            " * use OBJECT -- will only work with items in your inventory.\n\n"
            " * drop OBJECT -- drops an object from your inventory.\n\n"
            " * talk PERSON -- talk to a person in the current room. \n\n"
            " * inventory -- prints your inventory.\n\n"
            " * observe -- shows a list of items in the room. Useful if you "
            " want to pick something up not explicitly in the description.\n\n"
            " * fight -- asks opponent, what you want to do, and what weapon "
            "from your inventory (or fists, foot, etc...) you want to use.\n\n"
            " * save -- saves the game.\n\n"
            " * regenerate -- changes the description of the current room to a "
            "new one. This is mainly used for fixing descriptions that don't "
            "make sense.\n\n"
            " * quit -- quits the game.\n\n"
            "Each location records what has happened at that location and uses "
            "that record to decide what happens next. You can, however, bring "
            "objects from one location to another.\n\n\n\n" )
        print(msg)
        input_load_flag = input("Would you like to load a game? (Y or N) >>>")
        if input_load_flag == "Y" or input_load_flag == "y":
### load game           
            interactive_flag = 1
            print("please choose:")
            files=[]            
            for file in os.listdir("src/"):
                 if file.endswith(".pkl"):
                     print(file[:-4])
                     files.append(file)
            filename=""
            while not (filename in files):
                filename=input("please check spelling >>> ")
                if filename[-4:] != ".pkl":
                    filename = filename + ".pkl"
            file = open("src/" + filename, 'rb')
            data = pickle.load(file)
            descriptions=data[0]
            rooms=data[1] 
            room_connections=data[2] 
            input_persona=data[3] 
            input_location=data[4]
            input_atmosphere=data[5]
            current_room = 0
            inventory = set()
            if len(data)>6:
                inventory = set(data[6])
                current_room=data[7]
                
            
       
        else:
### start a new game without loading
            precalculate = input("Would you like to pregenerate room descriptions? It takes some time upfront, but gameplay is faster. (Y or N) >>> ")
            if precalculate in {"y","Y","yes"}:
                interactive_flag = 0
            else:
                interactive_flag = 1
            input_persona = input("Describe your character in a few words. You are >>> ")
            input_location = input("Describe where the adventure takes place in a few words.\n You would like to explore the >>> ")
            input_atmosphere = input("Describe the feeling of the place in a few adjectives, separated by commas. The " + input_location + " is >>> ")
            if input_location.startswith('the '):
                input_location=input_location[4:]
            if input_location.startswith('The '):
                input_location=input_location[4:]
            f=open("src/rooms.txt", "r", errors='ignore')
            if f.mode == 'r':
                contents =f.read()
            raw_text = contents + "\r\n" + input_atmosphere + " " + input_location + ":"
            print('generating places in the ' + input_location + '...')
            
            rooms=[]
            for _ in range(4 // batch_size):
                print("*", end =" ")
                text = description_gen.generate(raw_text)
                rooms = rooms + rooms_cleanup(text)           
            
            #remove duplicates from the list of rooms
            set_rooms=set(rooms)
            rooms = list(set_rooms)
            print(rooms)
            room_connections=create_graph(len(rooms),len(rooms)*3)
            print(room_connections)
            descriptions=[ '' for i in range(0,len(rooms)+1)]

        current_room = 0
        generate_more_flag = 0
        describe_flag = 1
        while True:
            
 ### main loop
            if "".__eq__(descriptions[current_room]):            
                # print("\n" + rooms[current_room] + "\n")
                #description_prompt = 'The following excerpt from a novel is a long and detailed description of the ' + input_atmosphere + ' things found in the ' + rooms[current_room] + ':\nYou are ' + input_persona + '. You are in the ' + rooms[current_room] + ' within the ' + input_location + '. Here is what you see there:'
                #description_prompt = 'The following excerpt from a novel is a long and detailed description of the ' + input_atmosphere + ' things found in the ' + rooms[current_room] + ':\nYou were ' + input_persona + '. You were in the ' + rooms[current_room] + ' within the ' + input_location + '. Here is what you saw there:'
                description_prompt = 'You were ' + input_persona + '. Previously, you were within a small stone room. Here is what you beheld: There were two chairs and a small table covered with various odds and ends. There was one small window and the floors were unswept. Later, you were in the ' + rooms[current_room] + ' of a ' + input_atmosphere + ' ' + input_location + '. You looked about at the furnishings of the ' + rooms[current_room] +'.'
               
                print("running description generator")
                text = description_gen.generate(description_prompt)
                descriptions[current_room] = description_cleanup(text)
            if describe_flag == 1:
                print("\n" + rooms[current_room] + "\n")
                wrap_print(descriptions[current_room])
                print("\n other nearby areas:", end = " ")
                describe_flag = 0
                for index in room_connections[current_room]:
                    print(rooms[index], end = " | ")
            
            if interactive_flag == 0:
                filename='src/' + input_location + '.pkl'
                afile = open(filename, 'wb')
                pickle_file=(descriptions,rooms, room_connections, input_persona, input_location, input_atmosphere, list(inventory), current_room)
                pickle.dump(pickle_file, afile)
                afile.close()
                describe_flag=1
                print("\n saved game.")
                
            
            if interactive_flag == 1:
                next_command = input("\n >>>")
                next_command_split = next_command.split(" ", 1)
                next_verb = next_command_split[0]
                next_object = 'none'
                if len(next_command_split)>1:
                    next_object = next_command_split[1]
                next_verb_past = getInflection(next_verb.strip(",.:-"), tag='VBD')
                if next_verb_past is None:
                    next_verb_past = "went"
                    next_verb = "go"
                    next_object ="ERROR"
                else:
                    next_verb_past=next_verb_past[0]
                
    
    ### verb handling
                if next_verb == "go":
                    describe_flag = 0
                    for i in range(0,len(rooms)):
                        if rooms[i].lower() == next_object.lower():
                            current_room = i
                            describe_flag = 1
                    if describe_flag == 0:
                        if next_object in descriptions[current_room]:
                            rooms.append(next_object)
                            room_list_length=len(rooms)
                            room_connections.append({current_room})
                            room_connections[current_room].add(room_list_length-1)
                            current_room = room_list_length-1
                            descriptions.append("")
                            describe_flag = 1
                        else:
                            wrap_print("Sorry, I don't recognize that. Please choose a place using a word from the description.")
    
                elif next_verb in {"get","grab"}: 
                    if next_object in inventory:
                        print("You already have that.")
                    elif next_object in descriptions[current_room]:
                        carrying_generator = carrying_prompt + "\n" + next_object + ":"
                        print("checking to see if you can get the object... ")
                        text = get_gen.generate(carrying_generator)
                        answer = text.split(" ")
                        if answer[1] == "okay":
                            print("You pick up the " + next_object + ".")
                            descriptions[current_room] = descriptions[current_room] + "You picked up the " + next_object + ". "
                            inventory.add(next_object)
                            print("inventory: ")
                            print(inventory)
                        elif answer[1] =="too":
                            print("that's too big to carry.")
                        else:
                            print("I don't know how to do that.")
                    else:
                        wrap_print("Sorry, I don't recognize that. Please choose an object to get using a word from the description.")
    
                elif next_verb == "save":
                    filename='src/' + input_location + '.pkl'
                    afile = open(filename, 'wb')
                    pickle_file=(descriptions,rooms, room_connections, input_persona, input_location, input_atmosphere, inventory, current_room)
                    pickle.dump(pickle_file, afile)
                    afile.close()
                    print("\n saved game.")
                    
                elif next_verb == "observe":
                    print("generating list of some objects found in this room...")
                    items = []
                    for _ in range(3 // batch_size):
                        text = description_gen.generate(items_prompt + "\n" + rooms[current_room] + "of" + input_location + ":")
                        items = items + rooms_cleanup(text)           
                    #remove duplicates from the list of items
                    set_items=set(items)
                    items = list(items)
                    comma_separated = ', '.join(items) 
                    outtext = "\nThe following objects are also in the room: " + comma_separated
                    wrap_print(outtext)
                    descriptions[current_room] = descriptions[current_room] + outtext
                
                elif next_verb == "talk":
                    partner = next_object
                    if partner in descriptions[current_room]:
                        is_animate = animate_prompt + "\n" + partner + ":"
                        text = get_gen.generate(is_animate)
                        animate_split = text.split("\n",1)
                        if animate_split[0] == " inanimate":
                            print("The " + partner + " just sit/s there.\n")
                            continue_chat = "n"
                        else:
                            print("Talking to " + next_object)
                            continue_chat = "y"
                            random_room = random.choice(rooms)
                            full_talk_prompt = 'If the '+partner+' wants '+input_persona+' to go to the library, the '+partner+' might say, "You really need to the library." If '+input_persona+' is supposed to go to the beach, the '+partner+' could say, "It is important for you to find the beach." If '+input_persona+' has to go to the '+random_room+', the '+partner+' would say something like, "'                            
                            while continue_chat == "y":
                                you_say = input("What do you say? (Just press enter to quit chat mode.) >>>")
                                if you_say == '':
                                    continue_chat = "n"
                                    descriptions[current_room] = descriptions[current_room] + '\nYou spoke with ' + partner + '.\n'
                                else:
                                    talk_prompt = full_talk_prompt + input_persona + ' says, "' + you_say + '"\n' + partner + ' says: "'
                                    #print("talk_prompt = " + talk_prompt)
                                    text = description_gen.generate(talk_prompt)
                                    #print("text =" + text)
                                    split_text = text.split('"')
                                    #print(split_text)
                                    response = split_text[0]
                                    print(next_object + ' says, "' + response + '"')
                                    full_talk_prompt = talk_prompt + response +'"\n'
                                    #print("full_talk_prompt = " + full_talk_prompt)
                    else:
                        print('try "talk PERSON" where "PERSON" is in the room description.')

    
                elif next_verb == "regenerate":
                    descriptions[current_room]=''
                    print("\n regenerating room description...")
                    describe_flag = 1
                    
                elif next_verb == "drop":
                    if next_object in inventory:
                        print("You drop the " + next_object + ".")
                        inventory.remove(next_object)
                        descriptions[current_room] = descriptions[current_room] + "You dropped the " + next_object + " here. "
                    else:
                        print("That item isn't in your inventory.")
                    
                elif next_verb == "use":
                    for item in inventory:
                        if next_object.startswith(item):
                            print("you use the " + next_object + ".")
                            descriptions[current_room] = descriptions[current_room] + "You used the " + next_object + ". "
                            generate_more_flag = 1
                    if generate_more_flag == 0:
                        print("You don't have that in your inventory.")
                        
                elif next_verb in {"fight", "punch", "stab", "attack", "kill"}:
                    
                    #check if the enemy is in the description
                    hp=[0,10,10]
                    enemy = input("opponent >>>")
                    if enemy in descriptions[current_room]:
                        continue_fight = "y"
                    else:
                        continue_fight = "n"
                        print("That opponent doesn't appear in the room description.")
                    if continue_fight == "y":
                        is_animate = animate_prompt + "\n" + enemy + ":"
                        text = get_gen.generate(is_animate)
                        animate_split = text.split("\n",1)
                        if animate_split[0] == " inanimate":
                            print("The " + enemy + " just sit/s there.\n")
                            continue_fight = "n"
                        
                   #start the "continue fight" loop
                    weapon = 'fists'
                    while continue_fight == "y":
                       #get the action
                       raw_action = input("action (e.g. stab) >>> ")
                       action_split = raw_action.split(" ", 1)
                       action_present = action_split[0]
                       action = getInflection(action_present, tag='VBD')
                       if action is None:
                            action = "hit"
                       else:
                            if action[0] is None:
                                action = "hit"
                            else:
                                action = action[0]
                       if action in {"quit","stopped","surrendered","hid","escaped","ran","fled"}:
                            print("You got away.")
                            continue_fight = "n"
                            break
                       #get the weapon and check it is in inventory
                       new_weapon = input("weapon (or press enter to use the same weapon) >>> ")
                       if new_weapon == '':
                            pass
                       else:
                            weapon = new_weapon
                       possible_weapons = inventory.union({"fists", "fist", "knee", "foot", "elbow", "head", "forehead", "finger", "fingers", "teeth", "voice", "hands", "hand", "feet", "knees", "elbows"})
                       for possible in possible_weapons:
                            if possible.endswith(weapon):
                                weapon = possible
                       if weapon in possible_weapons:
                           #generate response
                           start_sentence = "You " + action + " " + enemy + " with your " + weapon                    
                           prompt = "You are " + input_persona + ". Your adversary, " + enemy + ", faced off agaist you. You attacked with a mighty stroke, slicing " + enemy + "'s arm.\n" + enemy + " fought back, wounding your shoulder.\n You pressed your attack, wounding " + enemy + "'s leg.\n" + enemy + " tried again, but missed.\nYou pressed forward and took a mighty swing, but " + enemy + " escaped.\n" + enemy + " charged, dealing a heavy wound.\nYou managed to deal a nearly fatal blow, almost killing " + enemy + ".\n" + enemy + " let loose an onslaught, lightly wounding your arm.\nYou struck, but " + enemy + " got away unscathed.\n" + enemy + " retaliated with a barrage, doing heavy damage.\nYou fought back, rushing " + enemy + " and knocking " + enemy + " to the ground.\nYou rallied and caught " + enemy + " offguard.\n" + enemy + " blocked and returned the attack with a vicious strike.\nYou managed to get past " + enemy + "'s defenses and dealt a wound.\n" + enemy + " lunged, but missed by a mile.\nYou feinted to the left and struck to the right, but missed doing any damage.\n" + enemy + " knocked you off your feet with a heavy blow.\nYou fired away, successfully penetrating " + enemy + "'s defense.\n" + start_sentence
                           text = get_gen.generate(prompt)
                           text=other_cleanup(text)
                           text = start_sentence + text
                           sentences = text.split("\n")
                           paragraph_length = len(sentences)
                           sentences = sentences[0:min(paragraph_length,3)]
                           for sentence in sentences:
                                sentence=sentence.strip()
                                if sentence == '':
                                    sentences.remove(sentence)
                           # parse the response
                           for sentence in sentences:
                                #default is that no one is damaged 
                                damaged=0
                                # if the sentence starts with "you", then the enemy is the one damaged
                                yous = {'You', 'you'}                                
                                for term in yous:
                                    if sentence.startswith(term):
                                        damaged=1
                                for term in yous:
                                    # if the word "you" occurs somewhere else in the sentence, the player is the one damaged
                                        if term in sentence:
                                            if damaged == 0:
                                                damaged = 2
                                wases = {'was','were'}
                                swapflag = 0
                                # if the sentence is a passive sentence, then who is damaged gets swapped
                                for term in wases:
                                    if term in sentence:
                                        swapflag = 1
                                if swapflag == 1:
                                    if damaged == 2:
                                        damaged = 1
                                    else:
                                        damaged = 2
                                # damage_flag=1
                                # if someone is killed, the damage brings hitpoints to zero
                                kills = {"kills","killed","slay","slayed", "slays"}
                                for term in kills:
                                    if term in sentence:
                                        hp[damaged] = 0
                                        continue_fight = "n"
                                you_die = {"you die", "you are killed", "you are slain", "you are dead", "You die", "You are killed", "You are slain", "You are dead"}
                                for term in you_die:
                                    if term in sentence:
                                        hp[2] = 0
                                        continue_fight = "n"
                                dies = {"dies", "died", "die"}
                                for term in dies:
                                    if term in sentence:
                                        if hp[2] > 0:
                                            hp[1] = 0
                                            continue_fight = "n"  
                                #if a miss is mentioned, assume no damage was done
                                misses = {"escape","escaped","escapes","try", "tried", "tries", "miss", "missing", "missed", "misses", "dodge", "dodges", "dodged", "dodging", "block", "blocks", "blocked", "blocking", "save", "saved", "saving"}
                                for term in misses:
                                    if term in sentence:
                                        damaged = 0
                                #if the player or enemy are damaged, subtract one from their hitpoints
                                if damaged in {1,2}:        
                                    hp[damaged]=hp[damaged]-1
                                print(sentence)
                                print("enemy hp:", end=" ")
                                print(hp[1])
                                print("your hp:", end = " ")
                                print(hp[2])

                                if hp[1] < 1:
                                    print("ENEMY KILLED")
                                    continue_fight = "n"
                                    descriptions[current_room] = descriptions[current_room] + "\nOn the ground you see the remains of the " + enemy + "."
                                    break
                                if hp[2] < 1:
                                    print("YOU WERE KILLED (but we'll pretend you weren't so you can keep playing if you want)")
                                    continue_fight = "n"
                                    break
                       else:
                           print("You don't seem to have that weapon in your inventory.")
                            
                elif next_verb == "inventory": 
                    print("inventory:")
                    print(inventory)
                    
                elif next_verb == "quit":
                    raise SystemExit                  
                else:
                    generate_more_flag = 1
                
                #other verbs
                if generate_more_flag == 1:
                    generate_more_flag = 0                             
                    prompt = descriptions[current_room] + '\nYou ' + next_verb_past + " " + next_object
                    text=description_gen.generate(prompt)
                    text=other_cleanup(text)
                    wrapped = '\nYou ' + next_verb_past + " " + next_object + text
                    wrap_print(wrapped)
                    descriptions[current_room] = prompt  + text 
                    next_verb=""
            
### this steps through all the rooms generating them            
            if interactive_flag == 0:
                current_room=current_room + 1
                if current_room > len(rooms)-1:
                    current_room = 0
                    print("\n\n\n\n***** Pregeneration finished. Have fun! *****\n\n\n\n")
                    interactive_flag = 1


if __name__ == '__main__':
    fire.Fire(interact_model)
