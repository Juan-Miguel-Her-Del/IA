const assert=require('node:assert/strict');
const {CameraFlash}=require('../app/camera_component/camera_flash.js');

function camera(capability) {
  if(arguments.length===0)capability=true;
  return {
    readyState:'live',torch:false,calls:[],listeners:new Set(),
    getCapabilities(){return {torch:capability};},
    getSettings(){return {torch:this.torch};},
    getConstraints(){return {width:{ideal:1280},frameRate:{ideal:30},advanced:[{torch:false,zoom:1}]};},
    addEventListener(event,fn){this.listeners.add(fn);},
    removeEventListener(event,fn){this.listeners.delete(fn);},
    async applyConstraints(value){this.calls.push(value);this.torch=value.advanced.at(-1).torch;}
  };
}

async function main() {
  const flash=new CameraFlash();
  for(const capability of [undefined,false,[false],[true]]) {
    const track=camera(capability);flash.attach(track);
    assert.equal(flash.available,false);
    assert.equal(await flash.toggle(),false);
    assert.equal(track.calls.length,0);
  }
  for(const capability of [true,[false,true]]) {
    const track=camera(capability);flash.attach(track);
    assert.equal(flash.available,true);
    assert.equal(await flash.toggle(),true);assert.equal(flash.enabled,true);assert.equal(track.torch,true);
    assert.deepEqual(track.calls[0].width,{ideal:1280});assert.deepEqual(track.calls[0].frameRate,{ideal:30});
    assert.deepEqual(track.calls[0].advanced,[{zoom:1},{torch:true}]);
    assert.equal(await flash.toggle(),true);assert.equal(flash.enabled,false);assert.equal(track.torch,false);
  }
  const rejected=camera();rejected.applyConstraints=async()=>{throw new Error('Unsupported');};
  flash.attach(rejected);assert.equal(await flash.toggle(),false);
  assert.equal(flash.enabled,false);assert.equal(flash.busy,false);assert.ok(flash.error);
  assert.equal(rejected.readyState,'live','flash failure must not stop the camera');
  const ignored=camera();ignored.applyConstraints=async()=>{};
  flash.attach(ignored);assert.equal(await flash.toggle(),false);assert.equal(flash.enabled,false);
  let finish;
  const slow=camera();slow.applyConstraints=()=>new Promise(resolve=>{finish=resolve;});
  flash.attach(slow);const pending=flash.toggle();assert.equal(flash.busy,true);
  assert.equal(await flash.toggle(),false,'double clicks cannot enqueue toggles');
  const replacement=camera();flash.attach(replacement);finish();await pending;
  assert.equal(flash.track,replacement);assert.equal(flash.enabled,false);assert.equal(flash.busy,false);
  assert.equal(slow.listeners.size,0,'switching removes the old track listener');
  await flash.toggle();assert.equal(flash.enabled,true);
  replacement.readyState='ended';for(const fn of [...replacement.listeners])fn();
  assert.equal(flash.available,false);assert.equal(flash.enabled,false);assert.equal(flash.track,null);
  const broken=camera();broken.getCapabilities=()=>{throw new Error('No access');};
  flash.attach(broken);assert.equal(flash.available,false);
  flash.detach();assert.equal(flash.error,'');assert.equal(await flash.toggle(),false);
  console.log('Flash: support, on/off, preserved constraints, rejection, ignored settings, concurrent clicks, camera change and stop passed.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
