"""Read-only project status and source-model requirement checks; no simulation."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def inspect_requirements():
    state = read_json(ROOT / 'project.json')
    spec = read_json(ROOT / state['specification'])
    if not (ROOT / spec['requirements_source']).is_file():
        raise ValueError('User requirement document is missing')
    if spec['training']['initialization'] != 'random' or spec['training']['historical_models']:
        raise ValueError('Historical model initialization is outside the user requirements')
    if spec['training']['demo_updates_per_run'] != 1000:
        raise ValueError('Retain the current demo iteration budget of 1000 updates')
    delivery_budget = spec['training']['delivery_updates_per_run']
    if delivery_budget is not None and (type(delivery_budget) is not int or delivery_budget <= 0):
        raise ValueError('A configured delivery budget must be a positive integer')
    if spec['capability']['required_continuous_straight_speed_m_s'] != [-0.75, 0.75]:
        raise ValueError('Both continuous speed directions are required')
    if not spec['model']['read_only'] or spec['model']['physical_parameters_may_change']:
        raise ValueError('The source physical model is read-only')
    if not (ROOT / spec['model']['scene']).is_file():
        raise ValueError('Source scene is missing')
    model = ET.parse(ROOT / spec['model']['robot']).getroot()
    actuation = spec['actuation']
    radius = actuation['user_wheel_diameter_m'] / 2
    user_limit = actuation['user_max_straight_speed_m_s'] / radius
    wheels = {}
    for name in actuation['rear_wheel_actuators']:
        element = model.find(f"actuator/velocity[@name='{name}']")
        if element is None:
            raise ValueError(f'Missing source velocity actuator: {name}')
        low, high = map(float, element.attrib['ctrlrange'].split())
        if not low < 0 < high:
            raise ValueError(f'Bidirectional wheel actuation is required: {name}')
        limit = min(user_limit, -low, high)
        if abs(limit - actuation['expected_wheel_target_limit_rad_s']) > 1e-9:
            raise ValueError(f'Source wheel limit needs review: {name}={limit}')
        wheels[name] = dict(target_limit_rad_s=limit,
                           source_effort_range_nm=list(map(float, element.attrib['forcerange'].split())))
    hips = {}
    for name in actuation['hip_joints']:
        element = model.find(f"worldbody//joint[@name='{name}']")
        if element is None:
            raise ValueError(f'Missing source hip joint: {name}')
        limits = list(map(float, element.attrib['range'].split()))
        if not limits[0] <= spec['control']['landing_default_common_hip_rad'] <= limits[1]:
            raise ValueError('Default landing hip angle is outside the model limits')
        hips[name] = limits
    return dict(requirements_consistent_with_source=True,
                user_speed_derived_wheel_limit_rad_s=user_limit, rear_wheels=wheels,
                hip_source_limits_rad=hips, training_implemented=state['training_implemented'],
                policy_validated=state['policy_validated'],
                scope='Specification and XML inspection only; no simulation or training performed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['status', 'check'], nargs='?', default='status')
    args = parser.parse_args()
    result = inspect_requirements() if args.command == 'check' else read_json(ROOT / 'project.json')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
