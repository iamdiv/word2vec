import numpy as np 
import re
import matplotlib.pyplot as plt

def word_embedding(vocab_size,emb_size):
    wrd_emb = np.random.randn(vocab_size,emb_size)*0.01
    return wrd_emb

def initialize_weight(vocab_size,emb_size):
    W = np.random.randn(vocab_size,emb_size) * 0.01
    return W    
def initialize_parameters(vocab_size,emb_size):
    word_emb = word_embedding(vocab_size,emb_size)
    W = initialize_weight(vocab_size,emb_size)
    parameters = {}
    parameters["WRD_EMB"]= word_emb
    parameters["weights"]= W
    return parameters
def tokenize(text):
    
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())
def encode(tokens):
    
    word_to_id = {}
    id_to_word = {}
    for i in range(len(tokens)):
        word_to_id[tokens[i]] = i
        id_to_word[i] = tokens[i]
    return word_to_id,id_to_word
def generate_training(tokens,word_to_id,window_size):
    N = len(tokens)
    X,Y = [],[]
    for i in range(N):
        nbr_inds = list(range(max(0, i - window_size), i)) + \
                   list(range(i + 1, min(N, i + window_size + 1)))
        
        for j in nbr_inds:
            X.append(word_to_id[tokens[i]])
            Y.append(word_to_id[tokens[j]])
    X = np.array(X)
    X = np.expand_dims(X, axis=0)
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=0)
            
    return X, Y
def softmax(z):
    output = np.exp(z)/np.sum(np.exp(z),axis = 0,keepdims=True) 
    return output
def Forword_propagation(inds,parameters):
    m = inds.shape[1]
    wrd_emb = parameters["WRD_EMB"]
    word_vec = wrd_emb[inds.flatten(),:].T
    #assert(word_vec.shape == (word_vec.shape[1],m))
    W = parameters["weights"]
    Z = np.dot(W,word_vec)
    x = word_vec.shape[1]
    assert(Z.shape == (W.shape[0],x))
    output = softmax(Z)
    caches = {}
    caches['inds'] = inds
    caches['WRD_VEC'] = word_vec
    caches["weights"] = W
    caches["Z"] = Z
    return output,caches
def softmax_backward(Y,output):
    dl_dz = output - Y
    assert(dl_dz.shape == output.shape)
    return dl_dz
def dence_backward(dl_dz,caches):
    W = caches["weights"]
    wrd_vec = caches["WRD_VEC"]
    m = wrd_vec.shape[1]
    dl_dw = (1/m) * (np.dot(dl_dz,wrd_vec.T))
    dl_dwrd_vec = np.dot(W.T,dl_dz)
    return dl_dw,dl_dwrd_vec

def Backword_propagation(Y,output,caches):
    dl_dz = softmax_backward(Y,output)
    dl_dw,dl_wrd_vec = dence_backward(dl_dz,caches)
    gradient = {}
    gradient["dl_dz"] = dl_dz
    gradient["dl_dw"] = dl_dw
    gradient["dl_dwrd_vec"] = dl_wrd_vec
    return gradient
def update_parameters(parameters,caches,gradients,learning_rate):
    vocab_size,emb_size = parameters['WRD_EMB'].shape
    inds = caches['inds']
    wrd_emb = parameters['WRD_EMB']
    dl_dwrd_vec = gradients['dl_dwrd_vec']
    m = inds.shape[-1]
    wrd_emb[inds.flatten(), :] -= dl_dwrd_vec.T * learning_rate
    parameters["weights"] -= learning_rate*gradients['dl_dw']
def cross_entropy(output,Y):
    m = output.shape[1]
    cost = -(1 / m) * np.sum(np.sum(Y * np.log(output + 0.001), axis=0, keepdims=True), axis=1)
    return cost


def model_training(X,Y,vocab_size,emb_size,learning_rate,epochs,batch_size = 256,parameters=None,print_cost= True,plot_cost=True):
    cost = []
    m = X.shape[1]
    if parameters == None:
        parameters = initialize_parameters(vocab_size,emb_size)
    for epoch in range(epochs):
        epoch_cost = 0
        batch_inds = list(range(0,m,batch_size))
        np.random.shuffle(batch_inds)
        for i in batch_inds:
            batch_x = X[:,i:i+batch_size]
            batch_y = Y[:,i:i+batch_size]
            output,caches = Forword_propagation(batch_x,parameters)
            gradients = Backword_propagation(batch_y,output,caches)
            update_parameters(parameters,caches,gradients,learning_rate)
            cross = cross_entropy(output,batch_y)
            epoch_cost += np.squeeze(cross)
        cost.append(epoch_cost)
        if print_cost and epoch % (epochs // 500) == 0:
            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
        if epoch % (epochs // 100) == 0:
            learning_rate *= 0.98
            
    if plot_cost:
        plt.plot(np.arange(epochs), cost)
        plt.xlabel('# of epochs')
        plt.ylabel('cost')
    plt.show()
    return parameters



doc = "After the deduction of the costs of investing, " \
      "beating the stock market is a loser's game."
tokens  = tokenize(doc)
word_to_id,id_to_word = encode(tokens)
print(tokens)
print(word_to_id,id_to_word)
vocab_size = len(id_to_word)
window_size = 3
X,Y = generate_training(tokens,word_to_id,window_size)
m = Y.shape[1]
Y_one_hot = np.zeros((vocab_size,m))
Y_one_hot[Y.flatten(), np.arange(m)] = 1
parameters = model_training(X, Y_one_hot, vocab_size, 50, 0.05, 5000, batch_size=128, parameters=None, print_cost=True)

