import numpy as np


def human_feedback(keyboard_obs, action, feedback_type):
    if feedback_type == "evaluative":
        feedback = keyboard_obs.get_label()

    elif feedback_type == "dagger":
        if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
            action = correct_action(keyboard_obs, action)
            feedback = -1  # corrected
        else:
            feedback = 0  # bad

    elif feedback_type == "iwr":
        if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
            action = correct_action(keyboard_obs, action)
            feedback = -1  # corrected
        else:
            feedback = 1  # good

    elif feedback_type == "ceiling_full":
        if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
            action = correct_action(keyboard_obs, action)
            feedback = -1  # corrected
        else:
            feedback = keyboard_obs.get_label()

    elif feedback_type == "ceiling_partial":
        if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
            action = correct_action(keyboard_obs, action, full_control=False)
            feedback = -1  # corrected
        else:
            feedback = keyboard_obs.get_label()

    else:
        raise NotImplementedError("Feedback type not supported!")
    return action, feedback


def correct_action(keyboard_obs, action, full_control=True):
    if full_control:
        action[:-1] = keyboard_obs.get_ee_action()
    elif keyboard_obs.has_joints_cor():
        ee_step = keyboard_obs.get_ee_action()
        action[:-1] = action[:-1] * 0.5 + ee_step
        action = np.clip(action, -0.9, 0.9)
    if keyboard_obs.has_gripper_update():
        action[-1] = keyboard_obs.get_gripper()
    return action
