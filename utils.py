import itertools
import numpy as np
from numpy.linalg import norm
import pandas as pd
import itertools
from collections import Counter
import codecs


def zipf(word,wordcounts,d1=None,d2=None):
    '''
    Compute zipf frequency measure for given work w.r.t given corpus wordcounts
    :param word: string
    :param wordcounts: dictionary with token occurrencies in corpus (e.g. collections.Counter)
    :param d1: float with total number of token in corpus in millions
    :param d2: float with number of unique tokes in corpus in millions
    '''
    n = (wordcounts[word] + 1)
    if d1 is None:
        d1 = sum(wordcounts.values())/1000000
    if d2 is None:
        d2 = len(wordcounts.keys())/1000000

    return np.log10( n/(d1+d2) ) + 3

def corresp_transpose(word, mod_a, mod_b, reverse=False):
    if reverse:
        mod_a, mod_b = mod_b ,mod_a
    try:
        assert word in mod_a.wv.vocab.keys()
    except:
        return None
    return mod_b.wv.most_similar([mod_a.wv[word] ], topn=1 )[0][0]

def augment_single(word, model, n=1, m=1000, sample=True):
    '''Data augmentation inside single model'''
    synth = []
    seed = model.wv[word]
    for _ in range(n):
        p = []
        curr = seed
        for _ in range(m):
            step = np.random.normal(size=(model.vector_size))
            step /= norm(step)

            curr = curr + step

            if model.wv.most_similar([curr], topn=1)[0][0] == word:
                p.append(curr)
            else:
                break
        if sample: p = p[np.random.choice(range(len(p)))]
        synth.append(p)
    return np.array(synth)


def augment_multi(word, models, n=1, m=1000, sample=True):
    '''Data augmentation across multiple models'''
    synth = []
    seed = np.mean(np.vstack( [mods.wv[word] for mods in models]),axis=0)
    for _ in range(n):
        p = []
        curr = seed
        for _ in range(m):
            step = np.random.normal(size=(models[0].vector_size))
            step /= norm(step)

            curr = curr + step

            if all([mods.wv.most_similar([curr], topn=1)[0][0] == word for mods in models]):
                p.append(curr)
            else:
                break
        if sample: p = p[np.random.choice(range(len(p)))]
        synth.append(p)

    return np.array(synth)


def vocabulary_dataframe(corpora, verbose=False, filter_common = True, min_count=5, encoding="utf-8"):
    '''Build zipf dataframe for corpora'''
    # build zipf dicts
    wordzipfers = {}
    for f in corpora:
        name = f.split("/")[-1][:-4]
        with codecs.open(f, encoding=encoding) as fin:
            wc = Counter(fin.read().split())
        if verbose: print("count",name)
        
        # filter on min_count frequency
        if min_count:
            wc = {x : wc[x] for x in wc if wc[x] >= min_count}
            if verbose: print("min_count",name)

        
        d1 = sum(wc.values())/1000000
        d2 = len(wc.keys())/1000000
        wordzipfers[name] = { word:zipf(word,wc,d1,d2) for word in wc.keys()}
        if verbose: print("zipf",name)
        del wc
        
    # build dataframe of zipfs
    df = pd.DataFrame({"zipf %s" % (k) :v for k,v in wordzipfers.items()})

    if verbose: print("dataframe done")    
    
    # filter on shared vocab
    if filter_common:
        shared = None
        for k in wordzipfers.keys():
            if not shared:
                shared = set(wordzipfers[k].keys())
            else:
                shared = shared.intersection(set(wordzipfers[k].keys()))
        df = df.filter(shared,axis=0)
        if verbose: print("common filter done")
    
    return df


def comparative_dataframe(models, corpora, min_count = 5, filter_common = True, verbose = True, encoding="utf-8"):
    ''' Build comparative dataframe for multiple models'''
    names = [f.split("/")[-1][:-4] for f in corpora]

    if len(models)!=len(corpora):
        raise RuntimeError("models and corpora must be same length")

    df = vocabulary_dataframe(corpora, verbose=verbose, filter_common = filter_common, min_count=min_count, encoding=encoding)

    for i, j in itertools.permutations(range(len(models)),2):
        df['Corr %s2%s' %(names[i],names[j])] = [ corresp_transpose(x,models[i],models[j]) for x in df.index ]
        df['Corr %s2%s' %(names[j],names[i])] = [ corresp_transpose(x, models[i],models[j],reverse=True) for x in df.index ]

    return df


def lexicon_refinement(lex, models, corpora, score_center = 0, zipf_cutoff=5, verbose=False, comp_df=None):
    ''' Create stable lexicon dataframe from lexicon dictionary, aligned models and corpora
    '''
    if len(models) != len(corpora): raise RuntimeError("%s != %s: models and corpora must be same length" % (len(models),len(corpora)) )
    
    if comp_df is not None:
        df = comp_df
    else:
        df = comparative_dataframe(models, corpora)
    
    df['Valence'] = pd.Series(lex) - score_center
    df_val = df.filter(lex.keys(),axis=0)
    
    filter_zipf = np.logical_and( df_val.iloc[:,0] > zipf_cutoff, df_val.iloc[:,1] > zipf_cutoff)
    filter_corr = np.logical_and( 
                    df_val.iloc[:,2] == df_val.iloc[:,3],
                    df_val.index == df_val.iloc[:,2])
    return df_val[filter_zipf & filter_corr].Valence


def enrich(lex, models, n_target=None, verbose=True, return_words=False, msteps=200):
    ''' Balance lexicon through data augmentation
    '''
    if not len(set( [m.wv.vector_size for m in models] )) == 1: raise RuntimeError("Models MUST have same vector_size")
    
    dims = models[0].wv.vector_size
    
    # how many IN TOTAL @ end
    if n_target is None:
        n_target = int((2*dims)*1.2)
    
    word_pos = [k for k in lex.keys() if lex[k] > 0 ]
    word_neg = [k for k in lex.keys() if lex[k] < 0 ]
    word_full = word_pos + word_neg
    
    vect_original = [np.mean( [models[0].wv[w], models[1].wv[w] ] , axis=0 ) for w in word_full ]
    vect_pos = vect_original[:len(word_pos)]
    vect_neg = vect_original[len(word_pos):]
    
    labs = [lex[w] for w in word_full]
    labs_pos = labs[:len(word_pos)]
    labs_neg = labs[len(word_pos):]
    
    word_new = []
    vect_new = []
    labs_new = []
    
    i = 0
    
    # balancing
    if len(word_pos) != len(word_neg):
        
        word_minor = word_pos if len(word_pos)< len(word_neg) else word_neg
        vect_minor = vect_pos if len(word_pos)< len(word_neg) else vect_neg
        #l_minor = +1 if len(word_pos)< len(word_neg) else -1
        l_minor = labs_pos if len(word_pos)< len(word_neg) else labs_neg
        
        delta = max(len(word_pos), len(word_neg)) - len(word_minor)
        
        if verbose: print("Balancing: ", delta)
        
        for _ in tqdm.tqdm(range( delta )):

            idx = np.random.choice(range(len(word_minor)))
            
            w = word_minor[idx]
            v = augment_multi(w,models,m=msteps)[0]
            l = l_minor[idx]
            
            word_new.append(w+"_%s" % i)
            vect_new.append(v)
            labs_new.append(l)
            
            i+=1
    
    word_balanced = word_full + word_new
    labs_balanced = np.append(labs, labs_new)
    
    assert sum([x > 0 for x in labs_balanced ] ) == sum([x < 0 for x in labs_balanced ] )
    
    # growth
    if len( word_balanced ) < n_target:
        delta = n_target - len( word_balanced )
        
        if verbose: print("Growing: ", delta)
        
        for _ in tqdm.tqdm(range(delta)):
            idx = np.random.choice(range(len(word_balanced)))
            
            w = word_balanced[idx].split("_")[0]
            v = augment_multi(w,models,m=msteps)[0]
            l = labs_balanced[idx]

            
            word_new.append(w+"_%s" % i)
            vect_new.append(v)
            labs_new.append(l)
            i+=1
    
    if return_words:
        return np.append(vect_original,vect_new,axis=0), np.append(labs, labs_new), word_pos + word_neg + word_new
    else:
        return np.append(vect_original,vect_new,axis=0), np.append(labs, labs_new)
