import progressbar
class Solver(object):
    def __init__(self, model, **kwargs):
        self.model = model
        self.num_epochs = kwargs.pop('num_epochs', None)
        self.batch_size = kwargs.pop('batch_size', None)
        self.learning_rate = kwargs.pop('learning_rate', None)
        self.loss_fn = kwargs.pop('loss_fn', None)
        self.verbose = kwargs.pop('verbose', None)
        self.print_every = kwargs.pop('print_every', None)
        
    
    def build_network(self, inputs, labels):
        outputs = self.model(inputs)
        self.loss = self.loss_fn(
                labels=labels,
                predictions=outputs)

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
 
    
    def train(self, dataset):
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        labels = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        
        self.build_network(inputs, labels)
        bar = progressbar.ProgressBar(max_value=dataset.num_batches)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for epoch in range(self.num_epochs):
            for batch in range(dataset.num_batches):
                inputs_, labels_ = dataset.get_next_batch()
                loss = sess.run(self.loss, feed_dict={inputs: inputs_, labels: labels_})
                sess.run(self.train_step, feed_dict={inputs: inputs_, labels: labels_})
                #print(loss)
                bar.update(batch+1, force=True)
            


