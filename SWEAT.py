import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class SWEAT:
    def __init__(self, model_1, model_2, A, B, names=None):
        self.model_1 = model_1
        self.model_2 = model_2
        self.A = A
        self.B = B
        
        self.names = {"X1":"X1",
                      "X2":"X2",
                      "A" : "A",
                      "B" : "B"}
        
        if names is not None:
            if type(names) != dict: 
                raise RuntimeError("Names parameter must be dictionary")
            if list(names.keys()) != ["X1","X2","A","B"]: 
                raise RuntimeError('Names dictionary keys must be ["X1","X2","A","B"] ')
            self.names = names

    def word_assoc(self, model, w, test=False):
        assocs_A = [model.wv.similarity(w, a) for a in self.A]
        assocs_B = [model.wv.similarity(w, b) for b in self.B]
        if test:
            return stats.ttest_rel(assocs_A,assocs_B)
        else:
            return np.mean(assocs_A) - np.mean(assocs_B)


    def test(self, X, n=10000, same=True, two_tails=True, verbose=False):
        """
        :param X:
        :param n:
        :param same:
        :param two_tails:
        :return:
        """

        if same:
            modscore_one = [self.word_assoc(self.model_1, x) for x in X]
            modscore_two = [self.word_assoc(self.model_2, x) for x in X]
            assoc_scores = modscore_one + modscore_two

            sum_0 = sum(modscore_one)
            sum_1 = sum(modscore_two)

            score = sum(modscore_one) - sum(modscore_two)
            
            pool_std = np.sqrt( (np.std(modscore_one)**2 + np.std(modscore_two)**2)/2  )
            eff_size = (np.mean(modscore_one) - np.mean(modscore_two)) / pool_std
            
            if abs(sum_0) < 1e-3: print("Warning: %s is neutral" % self.names['X1'])
            if abs(sum_1) < 1e-3: print("Warning: %s is neutral" % self.names['X2'])

            # permutation test
            ds = []
            for _ in range(n):
                np.random.shuffle(assoc_scores)
                ds.append(sum(assoc_scores[:len(X)]) - sum(assoc_scores[-len(X):]))

            if two_tails:
                over = sum([abs(d) <= abs(score) for d in ds])
            else:
                if score > 0:
                        over = sum([d <= score for d in ds])
                else:
                        over = sum([d >= score for d in ds])

            pval = 1 - (over / n)
            
            if verbose:
                if score <0:
                    print("%s ~ %s" % (self.names['X1'],self.names["B"]) )
                    print("%s ~ %s" % (self.names['X2'],self.names["A"]) )
                elif score >0:
                    print("%s ~ %s" %(self.names['X1'],self.names["A"]) )
                    print("%s ~ %s" %(self.names['X2'],self.names["B"]) )
            
            return {"score":round(score, 4), "eff_size":round(eff_size,4), "p-val" :round(pval, 4)}
        else:
            raise NotImplementedError

    def plot_details(self, X, names=None, inner_pval=None):
        """ Plot SWEAT associations for target terms X wrt polarization sets A&B for models slices
            - models: gensim models
            - X: target terms (strings)
            - names: dictionary for plot labels
        """

        if names is not None:
            if type(names) != dict:
                raise RuntimeError("Names argument must be dictionary")

        if inner_pval is not None:
            if type(inner_pval) != float:
                raise RuntimeError("Confidence Level argument must be float")

        # association multi-array
        # (models) x (topic words) x (pos, neg) x (polar word)
        assocs = [
            [
                [
                    [m.wv.similarity(w, a) for a in self.A],
                    [m.wv.similarity(w, b) for b in self.B]
                ] for w in X
                ] for m in [self.model_1, self.model_2]
            ]

        f, axes = plt.subplots(1, 2, sharey=True)
        f.set_size_inches(12, len(X)*0.8)

        # for both slice models
        for i, ass_mod in enumerate(assocs):

            S = []  # vector for computing cumulative sum of association deltas
            ax = axes[i]

            # for each word in topic-wordset
            for j, ass_word in enumerate(ass_mod):
                # get both attribute-wordsets
                assA = ass_word[0]
                assB = ass_word[1]

                # boxplots for attribute wordsets associations
                boxA = ax.boxplot(assA,
                                  positions=[2 * j - 0.3], widths=0.3,
                                  boxprops=dict(color="red"), vert=False, showmeans=True, meanline=True,
				  meanprops=dict(color="black",ls="-"),
				  medianprops=dict(lw=0))

                boxB = ax.boxplot(assB,
                                  positions=[2 * j + 0.3], widths=0.3,
                                  boxprops=dict(color="blue"), vert=False, showmeans=True, meanline=True,
                                  meanprops=dict(color="black",ls="-"),
                                  medianprops=dict(lw=0))

                # compute means and delta
                muA = np.mean(assA)
                muB = np.mean(assB)
                dAB = muA - muB
                S.append(dAB)

		# default arrow coloring
                arr_col = "blue" if (dAB < 0) else "red"

                # set gray coloring for non-significant deltas
                if inner_pval is not None:
                    pval = stats.ttest_rel(assA,assB)[1]
                    if pval > inner_pval: arr_col = "grey"

                # plot word arrow
                ax.arrow(muA, 2 * j, -dAB, 0,
                         head_width=0.15, head_length=0.02, lw=1.5, length_includes_head=True, color=arr_col
                         )

            # plot setup & cosmetics
            ax.set_yticks(list(range(0, 2 * len(X), 2)))
            ax.set_yticklabels(X)

            if names is None:
                labels = [self.names['A'], self.names['B']]
            else:
                labels = [names['A'], names['B']]
            ax.legend(handles=[boxA['boxes'][0], boxB['boxes'][0]], labels=labels)

            ax.axvline(0, lw=1, ls='--', alpha=0.3, color='k')
            ax.set_xlim(-1, 1)
            ax.set_xlabel('cosine similarity')
            if names is None:
                ax.set_title(self.names['X%s'%(i+1)])
            else:
                ax.set_title(names['models'][i])

        plt.show()

    def plot_cumulative(self, X, names=None):
        """ Plot cumulative SWEAT associations for target terms X
        """
        deltas = [
                    [
                        np.mean([m.wv.similarity(w, a) for a in self.A]) - np.mean([m.wv.similarity(w, b) for b in self.B])
                         for w in X
                    ] for m in [self.model_1, self.model_2]
                ]
        
        title=None
        attr_labels = [self.names["A"], self.names["B"]]
        mod_labels = [self.names["X1"],self.names["X2"]]
        bar_cols = ['#e15759','#4e79a7']
        dot_cols = ['black','white']
        
        if names is not None:
            if "Title" in names.keys(): title = names['Title']
            if "Attributes" in names.keys(): attr_labels = names['Attributes']
            if "Models" in names.keys(): mod_labels = names["Models"]
            if "Bar Colors" in names.keys(): bar_cols = names["Bar Colors"]
            if "Dot Colors" in names.keys(): dot_cols = names["Dot Colors"]
        
        plt.figure(figsize=(7,2))
        
        xl = 0
        
        for i, s_mod in enumerate(deltas):
            
            cumulative = sum(s_mod)
            
            pos = sum([x for x in s_mod if x > 0])
            neg = sum([x for x in s_mod if x < 0])

            plt.broken_barh([ (0,pos) ], yrange=(i-0.4, 0.8), facecolors=(bar_cols[0]) , label=attr_labels[0])
            plt.broken_barh([ (neg,abs(neg))], yrange=(i-0.4, 0.8), facecolors=(bar_cols[1]), label=attr_labels[1] )
            
            plt.scatter([cumulative],[i],facecolor=dot_cols[0],edgecolor=dot_cols[1], label='cumulate')
            
            xl = max(xl,max(abs(pos),abs(neg)))

            
        xl += xl/10
        plt.xlim(-xl,xl)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        
        plt.legend(handles=handles[:3],labels=labels[:3])
        plt.axvline(0,lw=1,color='k',alpha=0.5)
        plt.xlabel("Cumulative association")
        plt.yticks(range(len(deltas)),mod_labels)    
        if title is not None:
            plt.title(title)
        plt.show()
