/* Deterministic delayed-reply checks, without a physical camera or network. */
const assert = require('node:assert/strict');
const {trackFromReply, MAX_REPLY_AGE} = require('../app/camera_component/box_tracking.js');

function texture() {
  const width=96,height=72,pixels=new Uint8Array(width*height);
  let seed=72;
  for(let i=0;i<pixels.length;i++) {
    seed=(Math.imul(seed,1664525)+1013904223)>>>0;
    pixels[i]=seed>>>24;
  }
  return {pixels,width,height};
}
function shifted(frame,dx,dy) {
  const pixels=new Uint8Array(frame.pixels.length);
  for(let y=0;y<frame.height;y++)for(let x=0;x<frame.width;x++) {
    const sx=x-dx,sy=y-dy;
    if(sx>=0&&sx<frame.width&&sy>=0&&sy<frame.height)pixels[y*frame.width+x]=frame.pixels[sy*frame.width+sx];
  }
  return {...frame,pixels};
}
const frame=texture();
const detection={box:[.2,.2,.65,.8],track_id:1,material:null};
const request={sent:100,frame,frames:[frame]};
const delayed=trackFromReply(detection,request,1700);
assert.ok(delayed,'a visually verified stationary object survives a slow reply');
assert.equal(delayed.observedAt,1700);
assert.ok(delayed.isFresh(1701));
assert.equal(trackFromReply(detection,{...request,frames:[]},1700),null,'age alone cannot refresh a reply');
assert.equal(trackFromReply(detection,request,100+MAX_REPLY_AGE),null,'unbounded old replies are rejected');
const empty={...frame,pixels:new Uint8Array(frame.pixels.length)};
assert.equal(trackFromReply(detection,{...request,frames:[empty]},1700),null,'a removed object must not leave a ghost box');
assert.equal(trackFromReply(detection,{...request,frames:[frame,empty]},1700),null,'the newest frame must validate');
const lowTexture={...frame,pixels:new Uint8Array(frame.pixels.length).fill(100)};
assert.equal(trackFromReply(detection,{sent:100,frame:lowTexture,frames:[lowTexture]},900),null,'low texture cannot justify refreshing an expired reply');
const moved=trackFromReply(detection,{...request,frames:[shifted(frame,2,2),shifted(frame,3,2)]},1700);
assert.ok(moved,'motion compensation validates a moved object');
assert.ok(Math.abs(moved.box[0]-(detection.box[0]+3/95))<1e-6);
assert.ok(Math.abs(moved.box[1]-(detection.box[1]+2/71))<1e-6);
assert.equal(trackFromReply(null,request,150),null);
assert.equal(trackFromReply(detection,{...request,frame:null},150),null);
console.log('Camera replies: 9 scenarios passed, including delayed, removed, low-texture and moving objects.');
