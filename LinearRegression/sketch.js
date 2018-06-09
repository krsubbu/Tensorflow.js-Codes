let x_vals = [];
let y_vals = [];
let m, b;
let learningRate = 0.5;
let optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(600, 480);

  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function loss(pred, label) {
  return pred.sub(label).square().mean();
}

function predict(x) {
  //y=mx+b;
  const xs = tf.tensor1d(x);
  const ys = xs.mul(m).add(b);
  return ys;
}

function mousePressed() {
  x = map(mouseX, 0, width, 0, 1);
  y = map(mouseY, 0, height, 1, 0);
  x_vals.push(x);
  y_vals.push(y);
}

function draw() {
  background(50);
	tf.tidy(() => {
  if (x_vals.length > 0) {
    const ys = tf.tensor1d(y_vals);
    optimizer.minimize(() => loss(predict(x_vals), ys));
  }
});
  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let x = map(x_vals[i], 0, 1, 0, width);
    let y = map(y_vals[i], 0, 1, height, 0);
    point(x, y);
  }

tf.tidy(() =>{
  let lineX = [0, 1];
  let ys = predict(lineX);
  let lineY = ys.dataSync();

  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);
  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);
	strokeWeight(2);
  line(x1, y1, x2, y2);
	});

}
