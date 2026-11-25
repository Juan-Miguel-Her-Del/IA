"""Validation selection, immutable runtime freeze, and final test in distinct steps."""
import argparse
from training.intake_release import select_classifier, freeze_release, final_test, activate_release, rollback_release

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('select-classifier', 'freeze', 'test', 'activate', 'rollback'))
    p.add_argument('--run', required=True)
    p.add_argument('--resume-test', action='store_true', help='Recover interrupted evaluation with the exact frozen runtime.')
    p.add_argument('--defer-camera', action='store_true', help='Only with explicit user instruction to verify the camera later.')
    a = p.parse_args()
    if a.defer_camera:
        if a.phase != 'activate': p.error('--defer-camera applies only to activation.')
        activate_release(a.run, defer_camera=True)
    elif a.resume_test:
        if a.phase != 'test': p.error('--resume-test applies only to the frozen final evaluation.')
        final_test(a.run, resume=True)
    else:
        {'select-classifier': select_classifier, 'freeze': freeze_release, 'test': final_test,
         'activate': activate_release, 'rollback': rollback_release}[a.phase](a.run)
