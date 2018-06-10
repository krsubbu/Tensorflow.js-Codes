let x_vals = [];
let y_vals = [];
let a, b, c, d;
let learningRate = 0.5;
let optimizer = tf.train.adam(learningRate);

function setup() {
  createCanvas(600, 480);

  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
	d = tf.variable(tf.scalar(random(-1, 1)));

}

function loss(pred, label) {
  return pred.sub(label).square().mean();
}

function predict(x) {
  //y=mx+b;
  // y = ax^3+bx^2+cx+d;
  const xs = tf.tensor1d(x);
  const ys = xs.pow(tf.scalar(3)).mul(a).add(xs.square().mul(b)).add(xs.mul(c)).add(d);
  return ys;
}

function mouseDragged() {
  x = map(mouseX, 0, width, 0, 1);
  y = map(mouseY, 0, height, 1, 0);
  x_vals.push(x);
  y_vals.push(y);
}

function draw() {
  background(50);
	const curveX = [];
  for (let x = 0; x <1.2; x+= 0.01) {
    curveX.push(x);
  }

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

const ys = tf.tidy(()=>predict(curveX));
let curveY = ys.dataSync();
ys.dispose();

  tf.tidy(() => {
		beginShape();
		stroke(255);
		noFill();
		strokeWeight(1);
    for (let i = 0; i < curveX.length; i++) {
			let x = map(curveX[i],0,1,0,width);
      let y = map(curveY[i],0,1,height,0);
			vertex(x,y);
    }
		endShape();
  });
}
