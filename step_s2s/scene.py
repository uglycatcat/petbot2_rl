"""Read-only source MJCF, composed in memory with a finite 2 cm platform."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco


def source_fingerprint(path):
    sha=hashlib.sha256()
    for p in sorted(Path(path).rglob('*')):
        if p.is_file() and p.suffix.lower() in ('.xml','.stl'):
            sha.update(p.relative_to(path).as_posix().encode()+b'\0');sha.update(p.read_bytes())
    return sha.hexdigest()


def load_scene(model_dir,metadata):
    model_dir=Path(model_dir).resolve()
    if source_fingerprint(model_dir)!=metadata['source_model_sha256']:
        raise ValueError('Source model differs from the selected policy; refusing silent model changes')
    scene=ET.parse(model_dir/'mjcf/scene.xml').getroot()
    robot=ET.parse(model_dir/'mjcf/robot_description.xml').getroot()
    robot.find('compiler').set('meshdir',str(model_dir/'meshes'))
    for include in list(scene.findall('include')):scene.remove(include)
    for node in reversed(list(robot)):scene.insert(0,node)
    t=metadata['task']['terrain'];edge=metadata['asset']['initial_com'][0]+t['initial_distance_m']
    # Adding environment geometry does not mutate any robot body/geom/joint/actuator.
    world=scene.findall('worldbody')[-1]
    ET.SubElement(world,'geom',name='step_platform',type='box',
        pos=f"{edge+t['platform_length_m']/2} 0 {t['height_m']/2}",
        size=f"{t['platform_length_m']/2} {t['platform_width_m']/2} {t['height_m']/2}",
        rgba='.35 .55 .38 1',contype='1',conaffinity='3',friction='1.5 .01 .001')
    contact=scene.find('contact')
    if contact is not None:
        for pair in list(contact):
            if pair.get('geom1')=='floor':
                copy=ET.SubElement(contact,'pair',pair.attrib);copy.set('geom1','step_platform')
    model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
    return model,edge
