import os, sys
sys.path.append(os.getcwd())

import time
import pickle, dill
import argparse
import numpy as np
import tensorflow as tf

import filebasedutils as utils
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
import models

'''

$ python train.py -o "pretrained"

'''

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')
    
    parser.add_argument('--shard-data', '-sh',
                        default=None,
                        dest='shard_data',
                        help='Path to training shard folder. All files will be read. Large wordlists need to be broken up into many smaller files.')

    parser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')

    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    parser.add_argument('--iters', '-n',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length (default: 10)')
    
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator and discriminator (default: 128)')
    
    parser.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')
    
    parser.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda hyperparameter (default: 10)')
    
    return parser.parse_args()

args = parse_args()
print("loading data")

#reuse charmap if exists, this can take an insane amount of time to process.
### Dictionary
if os.path.exists(os.path.join(args.output_dir, 'charmap.pickle')) and os.path.exists(os.path.join(args.output_dir, 'charmap_inv.pickle')):
    print("Charmaps found, loading...")
    with open(os.path.join(args.output_dir, 'charmap.pickle'), 'rb') as f:
        charmap = pickle.load(f, encoding='latin1')

    # Reverse-Dictionary
    with open(os.path.join(args.output_dir, 'charmap_inv.pickle'), 'rb') as f:
        inv_charmap = pickle.load(f, encoding='latin1')

else:
    print("No charmaps found in target directory, generating.")
    lines, charmap, inv_charmap = utils.load_dataset(
        path=args.training_data,
        max_length=args.seq_length)

def splitter(wordline):
    # This is a string tensor coming in
    # Has to be called from tf.py_function to have access to the numpy *REQUIRED*
    # convert to simple python string:
    px = wordline.numpy().decode('utf-8')
    #use a constructor to loop over the string into a list and return a new tensor
    #return tf.convert_to_tensor([char for char in px])

    t = tf.convert_to_tensor(list(str(px)), dtype=tf.string)
    return t

if args.shard_data:
    files = tf.data.Dataset.list_files(args.shard_data+"*")
    lines = files.interleave(lambda x: tf.data.TextLineDataset(x), cycle_length=20) #cyclelength is number of cuncurrent files to read
else:
    lines = tf.data.TextLineDataset(args.training_data)

lines = lines.map(lambda x: tf.py_function(func=splitter, inp=[x], Tout=tf.string)) #magic sauce to get pythonic string splitting
lines = lines.padded_batch(args.batch_size, padded_shapes=args.seq_length, padding_values="`")

lines = lines.prefetch(buffer_size=1)

#iterator = lines.make_one_shot_iterator()
iterator = lines.make_initializable_iterator()
#iterator.initializer()
#next_element = iterator.get_next()

######

#print(lines[1:20])

#lines need to be array of tuplets, each tuplet a word broken into a list padded with "`" to max length.
# eg ('k', 'o', 'o', 'l', 'k', 'a', 't', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`', '`')

print("pickeling...")
# Pickle to avoid encoding errors with json
with open(os.path.join(args.output_dir, 'charmap.pickle'), 'wb') as f:
    pickle.dump(charmap, f)
    f.close()

with open(os.path.join(args.output_dir, 'charmap_inv.pickle'), 'wb') as f:
    pickle.dump(inv_charmap, f)
    f.close()
    
print("Number of unique characters in dataset: {}".format(len(charmap)))

real_inputs_discrete = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))

fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

disc_real = models.Discriminator(real_inputs, args.seq_length, args.layer_dim, len(charmap))
disc_fake = models.Discriminator(fake_inputs, args.seq_length, args.layer_dim, len(charmap))

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

print("alpha mixing?")
# WGAN lipschitz-penalty
alpha = tf.random_uniform(
    shape=[args.batch_size,1,1],
    minval=0.,
    maxval=1.
)

print("generating params")
print("differences..")
differences = fake_inputs - real_inputs
print("interpolates..")
interpolates = real_inputs + (alpha*differences)
print("gradients")
gradients = tf.gradients(models.Discriminator(interpolates, args.seq_length, args.layer_dim, len(charmap)), [interpolates])[0]
print("slopes")
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
print("gradient penalty")
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += args.lamb * gradient_penalty

print("generator and discriminator params...")
gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

print("Adamoptimizers...")
print("train op")
gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
print("Disc op")
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
print("Adam done...")

#grab sample for initial ngram test
datapump = tf.Session()
datapump.run(iterator.initializer)
next_element = iterator.get_next()
g = []
for i in range(10):
    #sample for ngram baseline
    fl = datapump.run(next_element).tolist()
    for r in fl:
        g.append(tuple(l.decode('UTF-8') for l in r))
#print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#print(g)
#print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#validation_char_ngram_lms = [utils.NgramLanguageModel(i+1, g, tokenize=False) for i in range(4)]
# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
print("moedling language...")


if os.path.exists(os.path.join(args.output_dir, 'true_char_ngram_lms.pickle')):
    print("Model found, loading...")
    with open(os.path.join(args.output_dir, 'true_char_ngram_lms.pickle'), 'rb') as f:
        true_char_ngram_lms = dill.load(f, encoding='latin1')

else:
    print("No ngram model found in target directory, generating.")
    true_char_ngram_lms = [utils.FileNgramLanguageModel(i+1, args.training_data,  args.batch_size, tokenize=False) for i in range(4)]
    print("Saving char ngram pickle")
    with open(os.path.join(args.output_dir, 'true_char_ngram_lms.pickle'), 'wb') as f:
        dill.dump(true_char_ngram_lms, f)
        f.close()

print(true_char_ngram_lms)
print("validation car ngrams...")
#validation_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[:10*args.batch_size], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [utils.NgramLanguageModel(i+1, g, tokenize=False) for i in range(4)]

#print(g)

for i in range(4):
    print("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
#ug.. can we use a gpu for this??
print("modeling again with all dict?")
#true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]


# TensorFlow Session
with tf.Session() as session:
#with tf.distribute.MirroredStrategy() as session:
    

    # Time stamp
    localtime = time.asctime( time.localtime(time.time()) )
    print("Starting TensorFlow session...")
    print("Local current time :", localtime)
    
    # Start TensorFlow session...
    session.run(tf.global_variables_initializer())
    session.run(iterator.initializer)
    next_element = iterator.get_next() #Tf yelled at me to use this only outside training loops

    # Dataset iterator
    '''
    #old
    def inf_train_gen():
        print("shuffleling iterator")

        while True:
            np.random.shuffle(lines)
            for i in range(0, len(lines)-args.batch_size+1, args.batch_size):
                yield np.array(
                    [[charmap[c] for c in l] for l in lines[i:i+args.batch_size]],
                    dtype='int32'
                )
    '''
    def inf_train_gen():
        print("shuffleling iterator")
        while True:
            
            data =session.run(next_element).tolist()
            
            dd=[]

            for word in data:
                try:
                    dd.append([charmap[l.decode('UTF-8')] for l in word])
                except:
                    print(word)
                    print(charmap)
                    quit()
            yield dd

            #yield np.array(dd, dtype='int32')


    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    gen = inf_train_gen()
    print("HUZZAH")
    print(next(gen))
    print(next(gen))
    print(next(gen))
    print(next(gen))
    print("HUZZAH")

    for iteration in range(args.iters + 1):
        start_time = time.time()

        # Train generator
        print("Training Generator")
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        print("Training Critic")
        for i in range(args.critic_iters):
            _data = next(gen)
            #print(_data)
            try:
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_inputs_discrete:_data}
                )
            except:
                print("outtadata, reroll")
                session.run(iterator.initializer)
                
                
        print("making plots..")
        lib.plot.output_dir = args.output_dir
        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)

        # Output to text file after every 100 samples
        if iteration % 100 == 0 and iteration > 0:

            samples = []
            for i in range(10):
                samples.extend(generate_samples())

            for i in range(4):
                lm = utils.NgramLanguageModel(i+1, samples, tokenize=False)
                lib.plot.plot('js{}'.format(i+1), lm.js_with(true_char_ngram_lms[i]))

            with open(os.path.join(args.output_dir, 'samples', 'samples_{}.txt').format(iteration), 'w', encoding="utf-8") as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")

        if iteration % args.save_every == 0 and iteration > 0:
            model_saver = tf.train.Saver()
            model_saver.save(session, os.path.join(args.output_dir, 'checkpoints', 'checkpoint_{}.ckpt').format(iteration))
            print("{} / {} ({}%)".format(iteration, args.iters, iteration/args.iters*100.0 ))

        if iteration == args.iters:
            print("...Training done.")
        
        if iteration % 100 == 0:
            lib.plot.flush()

        lib.plot.tick()
# Time stamp
localtime = time.asctime( time.localtime(time.time()) )
print("Ending TensorFlow session.")
print("Local current time :", localtime)
