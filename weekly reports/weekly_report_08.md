Weekly Report
=============

**Submitted by** Anna Kovács

**Period** 03/31/2025 → 04/04/2025

Urgent!
-------

> Previous weekend's issue: Train MP4 again on the workstation because of another code running my training has stopped after a few hours

Reminder
--------

> 
>

Issues
------

> Train MP4 again on the workstation because of another code running my training has stopped after a few hours

Daily work
----------

### Monday
- Train again MP4 model with different beta values because it has stopped during the weekend
- Meeting with my internal supervisor

### Tuesday
- researching info about how to make a VAE adaptable -> MAIN QUESTION: how to choose the number of the layers
- Meeting with Ines about the deatils of the adaptable model

> 20x20 -> 2 layers and 140x140 -> 4 layers
>   - One study even notes that using “only 4 upsampling and 4 downsampling layers” can limit model capacity for very large images​, reinforcing that
>   4 is suitod ti mid-sized inpputs (64×64 up to ~128×128) - https://norma.ncirl.ie/6610/1/dnyaneshwarisudhirmahajan.pdf#:~:text=,to%20train%20a%20model
>   - in our case 4 layers are enough for 140x140 as well
> Mostly used for layer calculation: input_size / 2^L ≈ 4–8 where L ≈ log2(input_size) – 2 -> for 140 it would be 5 and also for some values belove as well
>   - want to be consistent: don't calculate more than 4 layers for smaller images
>
>   if img_size == 140:
>       return 4  # hardcoded
>   elif img_size < 140:
>       return min(4, int(log2(img_size)) - 2)
>   else:
>       return int(log2(img_size)) - 2  # allow >4
> test it with 80 and other image sizes -> upsample lunger cancer data and try with that

- realizing another difference between MP4 and MP2_64: size of output padding in the decoder

### Wednesday
- 

### Thursday
- 

### Friday
- 


Plans for next week
-------------------

- 

Requests and proposals
----------------------

> Ideas, suggestions, enquiries for complementary information or requests for the allocation of additional resources.

Meeting take-aways:
----------------------
>   if img_size == 140:
>       return 4  # hardcoded
>   elif img_size < 140:
>       return min(4, int(log2(img_size)) - 2)
>   else:
>       return int(log2(img_size)) - 2  # allow >4
- above calculation is too much hard-coding -> INSTEAD: linear interpolation between num of layers and log2(num of pixels)
