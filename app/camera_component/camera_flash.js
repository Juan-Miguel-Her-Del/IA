/* Control the active camera's continuous torch; no photo capture or server API. */
class CameraFlash {
  constructor(onChange=()=>{}) {
    this.onChange=onChange;
    this.track=null;
    this.available=false;
    this.enabled=false;
    this.busy=false;
    this.error='';
    this.generation=0;
    this.ended=null;
  }

  state() {
    return {available:this.available,enabled:this.enabled,busy:this.busy,error:this.error};
  }

  emit() { this.onChange(this.state()); }

  detach() {
    if(this.track && this.ended)this.track.removeEventListener?.('ended',this.ended);
    this.generation++;
    this.track=null;this.ended=null;
    this.available=false;this.enabled=false;this.busy=false;this.error='';
    this.emit();
  }

  attach(track) {
    this.detach();
    if(!track || track.readyState==='ended')return;
    this.track=track;
    try {
      const capability=track.getCapabilities?.().torch;
      this.available=typeof track.applyConstraints==='function' &&
        (capability===true || (Array.isArray(capability) && capability.includes(true) && capability.includes(false)));
      this.enabled=track.getSettings?.().torch===true;
    } catch (_) {
      this.available=false;
    }
    this.ended=()=>{if(this.track===track)this.detach();};
    track.addEventListener?.('ended',this.ended,{once:true});
    this.emit();
  }

  async toggle() {
    if(!this.available || this.busy || !this.track || this.track.readyState==='ended')return false;
    const track=this.track,generation=this.generation,wanted=!this.enabled;
    this.busy=true;this.error='';this.emit();
    try {
      // Preserve resolution, selected device and other camera settings.
      const {torch:oldTorch,...base}=track.getConstraints?.() || {};
      const advanced=(base.advanced || []).map(item=>{
        const {torch,...rest}=item;
        return rest;
      }).filter(item=>Object.keys(item).length);
      await track.applyConstraints({...base,advanced:[...advanced,{torch:wanted}]});
      if(this.track!==track || this.generation!==generation)return false;
      const actual=track.getSettings?.().torch;
      if(typeof actual==='boolean' && actual!==wanted)throw new Error('Torch was not applied');
      this.enabled=wanted;
      return true;
    } catch (_) {
      if(this.track===track && this.generation===generation) {
        this.error=wanted?'No se pudo encender el flash.':'No se pudo apagar el flash.';
      }
      return false;
    } finally {
      if(this.track===track && this.generation===generation){this.busy=false;this.emit();}
    }
  }
}

if(typeof module!=='undefined' && module.exports)module.exports={CameraFlash};
