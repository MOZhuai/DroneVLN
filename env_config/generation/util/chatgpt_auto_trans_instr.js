function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function simulateKeyInput() {
    var event = new KeyboardEvent('keydown', {
        key: 'Enter',
        code: 'Enter',
        keyCode: 13,
        which: 13,
        shiftKey: false
    });
    textarea.dispatchEvent(event);
}

async function auto_trans() {
    var textarea = document.getElementById('prompt-textarea');
    var button = textarea.nextElementSibling;
    var first_prompt = "";
    var head = "[prompt] ";
    var end = "\nplease forget all previous commands.";
    var instruction_list = [
        "fly to the stern of the ship",
        "fly to the left side of flower",
        "turn left and go to the right side of tree",
        "move foward to reach the back of anvil",
        "keep roughly straight to the right of apple",
        "walk about a quarter of a circle around the apple",
        "pass the right and reach the back side of hydrant",
        "pass the left and reach the back side of stone",
    ];
    for (var i = 0; i < instruction_list.length; i++) {
        instruction = instruction_list[i];
        if (instruction[instruction.length - 1] !== '.') {
            instruction += ".";
        }
        textarea.value = head + instruction + end;
        button.disabled = false;
        button.click();
        await sleep(5000);
    }
}

auto_trans();
