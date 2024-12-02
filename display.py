from typing import Union

import mido
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw


def draw_piano(width_white_key=13, width_black_key=8, height=50):
    # Determine the total width of the keyboard (88 keys)
    num_white_keys = 52
    total_width = num_white_keys * width_white_key
    
    # Create the image
    img = Image.new("RGB", (total_width, height), "white")
    draw = ImageDraw.Draw(img)
    
    # Black keys layout for an octave (repeated 7 times + one extra)
    black_key_positions = [1, 2, 4, 5, 6]  # 0-indexed relative to white keys
    
    # Draw white keys
    for i in range(num_white_keys):
        x_start = i * width_white_key
        draw.rectangle([x_start, 0, x_start + width_white_key - 1, height], outline="black", fill="white")
    
    # Draw black keys (shorter and narrower)
    for octave in range(8):  # 7 octaves
        for pos in black_key_positions:
            key_index = octave * 7 + pos - 5
            if key_index >= 88:
                break
            x_start = key_index * width_white_key - width_black_key // 2
            if x_start > 0:
                draw.rectangle([x_start, 0, x_start + width_black_key - 1, height * 0.6], fill="black")
    
    return np.array(img)[:, :, 0]


def midi_to_image(
    data: Union[str, NDArray],
    ticks_per_beat=16,
    bpm=None,
    duration=None,
    with_keybord=False,
    keyboard_offset=0,
    save_filename=None,
) -> NDArray:
    """
    Convert midi file into binary image

    Parameters
    ----------
    date : Union[str, NDArray]
        Midi file name or Midi binary data as array of shape (n_samples, 88)
    ticks_per_beat : int, optional
        Number of pixels per bit, by default 16
    bpm : int, optional
        bpm, for duration computation, by default None
    duration : int, optional
        duration in second to convert, by default None
    with_keyboard : bool
        whether to draw a keyboard, default to False
    keyboard_offset : int
        Black margin, default to 0
    save_filename : str, optional
        save the image if not None.
    Returns
    -------
    NDArray
        array of notes, shape (n_ticks, 88) if with_keyboard=False, (n_ticks, 676)
    """
    # Parse MIDI file
    if isinstance(data, str):
        mid = mido.MidiFile(data)
        if ticks_per_beat is None:
            ticks_per_beat = mid.ticks_per_beat

        bpm = bpm or 120  # Default to 120 BPM
        sec_per_tick = 60 / (bpm * ticks_per_beat)
        
        # Calculate total time and initialize piano roll
        total_time = sum(msg.time for msg in mid)
        
        max_time = int(total_time / sec_per_tick)
        if duration is None:
            piano_roll = np.zeros((88, (max_time + 1)), dtype=np.uint8)
        else:
            piano_roll = np.zeros((88, int(duration / sec_per_tick)), dtype=np.uint8)
    
        # Track active notes
        active_notes = {}

        # Iterate over MIDI messages
        time = 0
        for msg in mid:
            time += msg.time / sec_per_tick  # Time in seconds
            
            if msg.type == 'note_on' and msg.velocity > 0:  # Note starts
                key = msg.note - 21
                if 0 <= key < 88:
                    active_notes[key] = time  # Record when the note starts
            
            elif msg.type in ['note_off', 'note_on'] and msg.velocity == 0:  # Note ends
                key = msg.note - 21
                if key in active_notes:
                    start_time = active_notes.pop(key)
                    start_pixel = int(start_time)
                    end_pixel = int(time)
                    
                    # Fill the range of pixels for the note duration
                    if end_pixel >= piano_roll.shape[1]:
                        new_cols = end_pixel - piano_roll.shape[1] + 1
                        piano_roll = np.pad(piano_roll, ((0, 0), (0, new_cols)), mode='constant')
                    
                    piano_roll[key, start_pixel:end_pixel] = 255
            if duration is not None and time >= piano_roll.shape[1]:
                break
        
        if duration is not None:
            final_shape = (88, min(int(duration / sec_per_tick), max_time+1))
            piano_roll = piano_roll[:, :final_shape[1]]  # Truncate if oversized
            if piano_roll.shape[1] < final_shape[1]:  # Pad if undersized
                pad_width = final_shape[1] - piano_roll.shape[1]
                piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)), mode='constant')
        
        notes = piano_roll.T
    else:
        notes = data
    print(np.max(notes))
    notes = (np.array(notes)/np.max(notes)).astype(np.int8)
    
    if not with_keybord:
        return notes
    
    piano = draw_piano()
    if keyboard_offset:
        piano = np.vstack([piano, np.zeros((keyboard_offset, piano.shape[1]))])

    wider_notes = np.zeros((notes.shape[0], piano.shape[1]))
    
    width_white_key = 13

    n_c = 0
    for count in range(52):
        if (count+5)%7 in [0, 1, 3, 4, 5] and ((count+5)+6)%7 not in [0, 1, 3, 4, 5]:
            wider_notes[:, int(count*width_white_key+1) : int(count*width_white_key+1) + 7] = notes[:,[n_c]]
            n_c += 1
            if n_c < 88:
                wider_notes[:, int(count*width_white_key+9) : int(count*width_white_key+9) + 6] = notes[:,[n_c]]
            n_c += 1
        elif (count+5)%7 in [0, 1, 3, 4, 5] and ((count+5)+6)%7 in [0, 1, 3, 4, 5]:
            wider_notes[:, int(count*width_white_key+3) : int(count*width_white_key+3) + 7] = notes[:,[n_c]]
            n_c += 1
            if ((count+5)+1)%7 in [0, 1, 3, 4, 5]:
                wider_notes[:, int(count*width_white_key+11) : int(count*width_white_key+11) + 4] = notes[:,[n_c]]
                n_c += 1
            else:
                wider_notes[:, int(count*width_white_key+11) : int(count*width_white_key+11) + 6] = notes[:,[n_c]]
                n_c += 1
        else:
            wider_notes[:, int(count*width_white_key+5) : int(count*width_white_key+5) + 7] = notes[:,[n_c]]
            n_c += 1

    res = np.vstack([piano, wider_notes*255]).astype(np.uint8)
    if save_filename:        
        img = Image.fromarray(res, mode="L")
        img.save(save_filename)
    
    return Image.fromarray(res, mode="L")
    