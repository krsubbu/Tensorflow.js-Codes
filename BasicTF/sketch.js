function setup() {
  noCanvas();


  tf.tidy(() => {
    let values = [];
    for (let i = 0; i < 30; i++) {
      values[i] = random(255);
    }
    const shape = [2, 3, 5]
    const tense = tf.tensor3d(values, shape, 'float32');
    const vtense = tf.variable(tense);

    // console.log(vtense);
    // console.log(tense.toString());
    // console.log(tense.get(0,0,0));
  });

  console.log(tf.memory().numTensors);

}
