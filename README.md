# Infinite-Adventure
My own take on something like AI Dungeon.

Install instuctions:
I recommend using the huggingface version of this game. To use it, you should first install huggingface Transformers using these instructions:
https://huggingface.co/transformers/installation.html
The first time you use it, it will automatically download the 6GB neural network model GPT-2 into its cache directory. This can take a while.

The game has the following additional dependencies (besides GPT-2):
json, os, random, pickle, textwrap (which all come with most versions of python 3) and numpy, fire, tensorflow 1.14, pyinflect (which you'll need to install).
You can install most of these using "pip install".

Gameplay instructions:
 Infinite Adventure is primarily an exploration game, since there is no goal except ones you set yourself. You can type anything you want at the prompt, as long as it starts with a verb. A few verbs have special effects:

 * go LOCATION -- takes you to that place. If the place already has a description, it appears down below. If not, you can still go there if it appears in the description but it takes a minute to generate the new description.

 * get OBJECT -- allows you to pick up some objects that appear in the description. If they are too big or can't be picked up, the system will tell you.

 * use OBJECT -- will only work with items in your inventory.

 * drop OBJECT -- drops an object from your inventory.
 
 * talk PERSON -- speak with someone in the current room.
 
 * observe -- shows a list of items in the room. Useful if you want to pick something up not explicitly in the description.

 * inventory -- prints your inventory.

 * fight -- asks opponent, what you want to do and what weapon from your inventory (or fists, foot, etc...) you want to use.

 * save -- saves the game.

 * regenerate -- changes the description of the current room to a new one. This is mainly used for fixing descriptions that don't make sense.

 * quit -- quits the game.

Each location records what has happened at that location and uses that record to decide what happens next. You can, however, bring objects from one location to another.

About the game:
Obviously this is a really rough draft, but I think it shows a few things that are possible in this genre beyond what has already been done in AI Dungeon and without any finetuning training. It is very slow on CPU (about 60 seconds to generate a room description) and uses too much memory for most GPUs. See if you can come up with better prompts in carrying.txt, rooms.txt, or in the variable "description generator" or "prompt" in the "fight" case in InfiniteAdventure.py. I appreciate any bug reports.

I added a version of the main file that uses huggingface-transformers and the GPT-2 model trained on adventure games for the AI Dungeon 2 standalone.
