
#Data Parameters
GRID_COUNT = 100

#RNN Model Prarmeters
batch_size = 200
place_dim = GRID_COUNT*GRID_COUNT
time_dim=48
pl_d=50
time_k=50
text_k = 50
hidden_neurons=50
learning_rate=0.01
model_file_name = str(batch_size)+'_'+str(pl_d)+'_'+str(hidden_neurons)+'_'+str(learning_rate)
