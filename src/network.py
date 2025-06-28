import random
import numpy as np
import pickle
class Network :
    def __init__(self,sizes):
        self.sizes=sizes
        self.num_layers=len(sizes)
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x) for (x,y) in zip(sizes[:-1],sizes[1:])]

    
    def SGD (self, training_data, epochs, mini_batch_size, eta, test_data=None):
        for j in range (0,epochs,1) :
            random.shuffle(training_data)
            mini_batch=[]
            mini_batches=[]
            for i in range (len(training_data)) :
                mini_batch.append(training_data[i])
                if len(mini_batch)==mini_batch_size:
                    mini_batches.append(mini_batch)
                    mini_batch=[]
            for mini_batch in mini_batches :
                self.update_mini_batch(mini_batch,eta)
            
            if test_data :
                print(f"epoch{j}:{self.evaluate(test_data)}/{len(test_data)}")
            else:
                print(f"epoch{j} complete !")
    
    def update_mini_batch(self,mini_batch,eta):
        # ∇C = Σ ∇Cx over every x in mini_batch
        gr_b=[np.zeros(b.shape) for b in self.biases]
        gr_w=[np.zeros(w.shape) for w in self.weights]
        # ∇C = Σ ∇Cx over every x in mini_batch ∇C = Σ ∇Cbx + Σ ∇CWx
        for (x,y) in mini_batch:
            """(gr_b,gr_w) += self.backprop(x, y) !!!!!! WRONG 
                because we only add sub mult ... numpy elements not tuples and lists""" 
            gr_bx,gr_wx=self.backprop(x,y) 
            #∇C = Σ ∇Cbx + Σ ∇CWx
            for i in range(len(gr_b)): gr_b[i]+=gr_bx[i] #Σ ∇Cbx 
            for i in range (len(gr_w)): gr_w[i]+=gr_wx[i] #Σ ∇CWx
        for i in range (len(self.biases)):self.biases[i]-=(eta/len(mini_batch))*gr_b[i] #bi ← bi - 1/m(η * Σ∇Cbxi)=bi -η * [(Σ∇Cbxi)/m];bi is the biases vector of the layer
        for i in range (len(self.weights)):self.weights[i]-=(eta/len(mini_batch))*gr_w[i] #Wi ← bi - 1/m(η * Σ∇CWxi)=bi -η * [(Σ∇CWxi)/m];Wi is the weights matrix
    
    def feedforward(self,a):
        for w,b in zip(self.weights,self.biases):
            z=np.dot(w,a)+b
            a=sigmoid(z)
        
        return a

    def backprop(self,x,y):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        a=x
        activations=[x]
        zs=[]
        for w,b in zip(self.weights,self.biases):
            z=np.dot(w,a)+b
            zs.append(z)
            a=sigmoid(z)
            activations.append(a)
        
        delta=self.costderivative(activations[-1],y) * sigmoid_prime(zs[-1])
        nabla_b[-1]=delta 
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        
        for l in range (2,self.num_layers):
            z=zs[-l]
            delta=(np.dot(np.transpose(self.weights[-l+1]),delta)) * sigmoid_prime(z)
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)

            
    


    def costderivative(self,a,y):
        return (a-y)
        

    def evaluate(self,test_data):
        s=0
        for (x,y) in test_data :
            output_layer=self.feedforward(x)
            if np.argmax(output_layer)==y :
                s+=1
        return s
    
    #backprop alg is next chapter so i will just copy the function from the book for now and just get an overview by reading it
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)
    
    #this cost_derivative function is also from the book its related to the back prop alg 
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)
    #these two methods to save the trained network param is from chat (will come back to code it myself later)
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": self.weights,
                "biases": self.biases}
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        net = Network(data["sizes"])
        net.weights = data["weights"]
        net.biases = data["biases"]
        return net
    

    
        
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

    



