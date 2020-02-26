import collections
import numpy as np
import re

def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        print("NgramlaguageModel")
        print("samples to process: {}".format(len(samples)))
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            #print("this takes forever?")
            #print(ngrams)
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in range(len(sample)-n+1):
                yield sample[i:i+n]

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        # p is another NgramLanguageModel
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        # p is another NgramLanguageModel
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i**2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i**2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        # p is another NgramLanguageModel
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5*(kl_p_m + kl_q_m) / np.log(2)

class FileNgramLanguageModel(object):
    #Gnererate 
    def __init__(self, n, samples, max_length, tokenize=False):
        print("FileNgramlaguageModel")
        print("uSingFile: {}".format(samples))
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples
        samples = open(samples, 'r',  encoding="utf-8")
        self._n = n
        self._samples = samples
        self._max_length = max_length
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            #print("this takes forever?")
            print(ngram)
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1
            print(self._total_ngrams)

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            sample=self._samples.readline()
            if len(sample) < 2:
                print("RRRRRRRRRRRRRRRRRRRRRRRAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                return
            sample = sample[:-1]
            sample = tuple(sample)
            sample = (sample + ( ("`",)*(self._max_length-len(sample)) ) )
            #print("sample: {}".format(sample))
            for i in range(len(sample)-n+1):
                #print("rawsample!@#!#!")
                #print(sample[i:i+n])
                #print("rawsample!@#!#!")
                yield sample[i:i+n]
            """
        while x <=n:
            line=self._samples.readline()
            line = line[:-1]
            line = tuple(line)
            #line.line + ( ("`",)*(max_length-len(line)) ) 
            retsamp.append(line + ( ("`",)*(self._max_length-len(line)) ) )
            x+=1
            # lines.append(line + ( ("`",)*(max_length-len(line)) ) )
            """

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        # p is another NgramLanguageModel
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        # p is another NgramLanguageModel
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i**2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i**2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        # p is another NgramLanguageModel
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5*(kl_p_m + kl_q_m) / np.log(2)


def load_dataset(path, max_length, tokenize=False, max_vocab_size=2048):
    import collections
    print("Load dataset, might blowup")
    lines = []
    linecount = 0
    counts = collections.Counter()
    #counts = collections.Counter()
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line[:-1]
            line=tuple(line)
            if len(line) > max_length:
                line = line[:max_length]
                continue # don't include this sample, its too long

            # right pad with ` character
            bline = (line + ( ("`",)*(max_length-len(line)) ) )
            #print(bline)
            #Build our charcount on the fly, no loading of whole dictionary
            counts.update(bline)
            linecount += 1
            '''
            for char in line:
                counts.update(char)
                '''
    
    print("done loading")
    np.random.shuffle(lines) #what the shit is this for? Why? it's all used and randomly sampled later...

    print("counting chars..")
    #this can be done while loading
    
    fcounts = collections.Counter(char for line in lines for char in line) 
    print(fcounts)
    print("this is built on the fly")
    print(counts)

    charmap = {'unk':0}
    inv_charmap = ['unk']
    print("building charmap")

    #wot? Some weird mapping I won't mess with
    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    #this makes no sense for this purpose... when are we going to have a charmap over 2k & need to INDIVIDUALLY filter every charactor?
    # now we get 2 copies of the exploded wordlist, dafuq!
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    # for i in range(100):
    #     print filtered_lines[i]

    print("loaded {} lines in dataset".format(len(lines)))
    #return filtered_lines, charmap, inv_charmap
    print("LInes countd: {}".format(linecount))
    return lines, charmap, inv_charmap
