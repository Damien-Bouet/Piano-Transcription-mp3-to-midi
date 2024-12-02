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
    max_note_beats=8,
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
        
        # Determine piano roll size
        if duration is None:
            piano_roll = np.zeros((88, (max_time + 1)), dtype=np.uint8)
        else:
            piano_roll = np.zeros((88, int(duration / sec_per_tick)), dtype=np.uint8)
        
        # Track active notes
        active_notes = {}
        
        # Iterate over MIDI messages
        time = 0
        max_note_duration = max_note_beats * (60 / bpm) * ticks_per_beat
        
        for i, msg in enumerate(mid):
            time += msg.time / sec_per_tick  # Time in seconds
            
            # Check and truncate any existing notes that exceed max duration
            for key in list(active_notes.keys()):
                note_info = active_notes[key]
                note_duration = time - note_info['start_time']
                
                if max_note_duration is not None and note_duration > max_note_duration:
                    # Truncate the note
                    end_time = note_info['start_time'] + max_note_duration
                    start_pixel = int(note_info['start_time'])
                    end_pixel = int(end_time)
                    
                    # Extend piano roll if needed
                    if end_pixel >= piano_roll.shape[1]:
                        new_cols = end_pixel - piano_roll.shape[1] + 1
                        piano_roll = np.pad(piano_roll, ((0, 0), (0, new_cols)), mode='constant')
                    
                    # Fill the range of pixels for the note duration
                    piano_roll[key, start_pixel:end_pixel] = 255
                    
                    # Remove the note from active notes
                    del active_notes[key]
            
            if msg.type == 'note_on' and msg.velocity > 0:  # Note starts
                key = msg.note - 21
                if 0 <= key < 88:
                    # Record when the note starts
                    active_notes[key] = {'start_time': time}
            
            elif msg.type in ['note_off', 'note_on'] and msg.velocity == 0:  # Note ends
                key = msg.note - 21
                if key in active_notes:
                    note_info = active_notes.pop(key)
                    start_time = note_info['start_time']
                    
                    start_pixel = int(start_time)
                    end_pixel = int(time)
                    
                    # Extend piano roll if needed
                    if end_pixel >= piano_roll.shape[1]:
                        new_cols = end_pixel - piano_roll.shape[1] + 1
                        piano_roll = np.pad(piano_roll, ((0, 0), (0, new_cols)), mode='constant')
                    
                    # Fill the range of pixels for the note duration
                    piano_roll[key, start_pixel:end_pixel] = 255
            
            # Break if we've exceeded desired duration
            if duration is not None and time >= piano_roll.shape[1]:
                break
        
        # Final duration and shape adjustment
        if duration is not None:
            final_shape = (88, min(int(duration / sec_per_tick), max_time+1))
            piano_roll = piano_roll[:, :final_shape[1]]  # Truncate if oversized
            if piano_roll.shape[1] < final_shape[1]:  # Pad if undersized
                pad_width = final_shape[1] - piano_roll.shape[1]
                piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)), mode='constant')
        
        notes = piano_roll.T
    else:
        notes = data

    notes = (np.array(notes)/np.max(notes)).astype(np.int8)
    
    if not with_keybord:
        return notes
    
    piano = draw_piano()
    if keyboard_offset:
        piano = np.vstack([piano, np.zeros((keyboard_offset, piano.shape[1]))])

    wider_notes = get_wider_notes(notes, piano.shape[1])

    res = np.vstack([piano, wider_notes*255]).astype(np.uint8)
    if save_filename:        
        img = Image.fromarray(res, mode="L")
        img.save(save_filename)
    
    return Image.fromarray(res, mode="L")
    

def get_wider_notes(notes, width):
    wider_notes = np.zeros((notes.shape[0], width))
    
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

    return wider_notes


def midis_comparison(target, preds, keyboard_offset=0, save_filename=None, target_color=(0, 0.5, 1), preds_color=(1, 0.8, 0), correct_color=(0, 0.9, 0), ):

    notes1 = (np.array(target)/np.max(target)).astype(np.int8)
    notes2 = (np.array(preds)/np.max(preds)).astype(np.int8)
    
    piano = draw_piano()
    if keyboard_offset:
        piano = np.vstack([piano, np.zeros((keyboard_offset, piano.shape[1]))]).T

    wider_notes1 = get_wider_notes(notes1, piano.shape[1]).astype(int)
    wider_notes2 = get_wider_notes(notes2, piano.shape[1]).astype(int)

    print(wider_notes1.shape, wider_notes2.shape)

    result_image = np.zeros((wider_notes1.shape[0], wider_notes1.shape[1], 3), dtype=np.float32)

    # Where both are white (255): Green
    both_white = (wider_notes1 == 1) & (wider_notes2 == 1)
    result_image[both_white] = correct_color

    # Where only the first is white: Blue
    only_first_white = (wider_notes1 == 1) & (wider_notes2 != 1)
    result_image[only_first_white] = target_color

    # Where only the second is white: Red
    only_second_white = (wider_notes1 != 1) & (wider_notes2 == 1)
    result_image[only_second_white] = preds_color

    res = np.vstack([np.stack([piano]*3, axis=2), result_image*255]).astype(np.uint8)

    # Convert the result array to an image
    img = Image.fromarray(res, mode="RGB")
    img.save("merged_image.png")
    if save_filename:        
        img = Image.fromarray(res, mode="L")
        img.save(save_filename)
    
    return Image.fromarray(res, mode="RGB")