This is now available as a colab, thanks to Nick Walton: https://colab.research.google.com/drive/1csbT4Wk_OZI_rfOzS5VBZNXFM2GCAznu
But it may be buggy for the next couple of days until I get it all tested.

# Infinite-Adventure
My own take on something like AI Dungeon.

Install instuctions:
You first should install and get working GPT-2 from openAI (https://github.com/nshepperd/gpt-2). This requires Python 3.7 since Tensorflow won't work on Python 3.8 yet. Note that OpenAI has made some minor changes to their code since this version so you should use the version of GPT-2 I linked to.
You will need to download their 1558M model into the "models" subfolder of GPT-2:

python download_model.py 1558M

All of the files for infinite-adventure go in the "src" subfolder of that project.
The game can save game files to that directory, so make sure it has permissions to do so.
The game has the following additional dependencies (besides GPT-2):
json, os, random, pickle, (which all come with most versions of python 3) and numpy, fire, tensorflow 1.14, pyinflect (which you'll need to install).
You can install most of these using "pip install".

Gameplay instructions:
 Infinite Adventure is primarily an exploration game, since there is no goal except ones you set yourself. You can type anything you want at the prompt, as long as it starts with a verb. A few verbs have special effects:

 * go LOCATION -- takes you to that place. If the place already has a description, it appears down below. If not, you can still go there if it appears in the description but it takes a minute to generate the new description.

 * get OBJECT -- allows you to pick up some objects that appear in the description. If they are too big or can't be picked up, the system will tell you.

 * use OBJECT -- will only work with items in your inventory.

 * drop OBJECT -- drops an object from your inventory.

 * inventory -- prints your inventory.

 * fight OPPONENT -- asks what you want to do and what weapon from your inventory (or fists, foot, etc...) you want to use.

 * save -- saves the game.

 * regenerate -- changes the description of the current room to a new one. This is mainly used for fixing descriptions that don't make sense.

 * quit -- quits the game.

Each location records what has happened at that location and uses that record to decide what happens next. You can, however, bring objects from one location to another.

About the game:
Obviously this is a really rough draft, but I think it shows a few things that are possible in this genre beyond what has already been done in AI Dungeon and without any finetuning training. It is very slow on CPU (about 60 seconds to generate a room description) and uses too much memory for most GPUs. See if you can come up with better prompts in carrying.txt, combat.txt, rooms.txt, or in the variable "description generator" in InfiniteAdventure.py. I appreciate any bug reports.
