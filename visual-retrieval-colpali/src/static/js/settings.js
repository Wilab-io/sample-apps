document.addEventListener('DOMContentLoaded', initializeSettingsPage);

// HTMX after swap event to reinitialize everything after content updates
document.addEventListener('htmx:afterSwap', function(event) {
    if (event.detail.target.id === 'settings-content') {
        initializeSettingsPage();
    }
});

function initializeSettingsPage() {
    const questionsContainer = document.getElementById('questions-container');
    const addButton = document.getElementById('add-question');
    const rankerInputs = document.querySelectorAll('input[name="ranker"]');
    const connectionInputs = document.querySelectorAll('input[name="gemini_token"]');
    const apiKeyFile = document.querySelector('input[name="api_key_file"]');
    const applicationPackageInputs = document.querySelectorAll('input[name="tenant_name"], input[name="app_name"], textarea[name="schema"]');
    const promptTextarea = document.querySelector('textarea[name="prompt"]');

    // Initialize prompt textarea if it exists
    if (promptTextarea) {
        promptTextarea.setAttribute('data-original', promptTextarea.value);
        promptTextarea.addEventListener('input', updatePromptSaveButtonState);
        updatePromptSaveButtonState();
    }

    if (rankerInputs.length > 0) {
        const checkedInput = document.querySelector('input[name="ranker"]:checked');
        if (checkedInput) {
            rankerInputs.forEach(input => {
                input.setAttribute('data-original', checkedInput.value);
                input.addEventListener('change', updateRankerSaveButtonState);
            });
        }
        updateRankerSaveButtonState();
    }

    // Store original values when page loads
    connectionInputs.forEach(input => {
        input.setAttribute('data-original', input.value);
        input.addEventListener('input', updateConnectionSaveButtonState);
    });

    if (apiKeyFile) {
        apiKeyFile.addEventListener('change', updateConnectionSaveButtonState);
    }

    if (connectionInputs.length > 0) {
        updateConnectionSaveButtonState();
    }

    applicationPackageInputs.forEach(input => {
        input.setAttribute('data-original', input.value);
        input.addEventListener('input', updateApplicationPackageSaveButtonState);
    });

    if (applicationPackageInputs.length > 0) {
        updateApplicationPackageSaveButtonState();
    }

    if (questionsContainer) {
        questionsContainer.addEventListener('input', function(e) {
            if (e.target.tagName === 'INPUT') {
                updateSaveButtonState();
            }
        });
    }

    if (addButton) {
        addButton.addEventListener('click', function() {
            const newQuestionDiv = document.createElement('div');
            newQuestionDiv.className = 'flex items-center mb-2';

            const input = document.createElement('input');
            input.className = 'flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background';
            input.name = `question_${questionsContainer.children.length}`;

            const deleteButton = document.createElement('button');
            deleteButton.className = 'delete-question ml-2';
            deleteButton.setAttribute('variant', 'ghost');
            deleteButton.setAttribute('size', 'icon');
            deleteButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trash-2">
                    <path d="M3 6h18"></path>
                    <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                    <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
                    <line x1="10" y1="11" x2="10" y2="17"></line>
                    <line x1="14" y1="11" x2="14" y2="17"></line>
                </svg>
            `;

            newQuestionDiv.appendChild(input);
            newQuestionDiv.appendChild(deleteButton);
            questionsContainer.appendChild(newQuestionDiv);

            updateSaveButtonState();
            input.focus();
        });
    }

    document.addEventListener('click', function(e) {
        if (e.target.closest('.delete-question')) {
            const questionDiv = e.target.closest('.flex');

            // Always allow deletion if it's not the first question
            if (!questionDiv.querySelector('input').name.endsWith('_0')) {
                questionDiv.remove();
                updateInputNames();
                updateSaveButtonState(true);
            }
        }
    });

    function updateInputNames() {
        const inputs = questionsContainer.querySelectorAll('input');
        inputs.forEach((input, index) => {
            input.name = `question_${index}`;
        });
    }

    function updateSaveButtonState(is_deletion = false) {
        const inputs = questionsContainer.querySelectorAll('input');
        const hasValidQuestion = Array.from(inputs).some(input => input.value.trim() !== '');
        var hasChanges = false;
        if (is_deletion) {
            hasChanges = true;
        } else {
            hasChanges = Array.from(inputs).some(input => {
                const originalValue = input.getAttribute('data-original') || '';
                return input.value.trim() !== originalValue.trim();
            });
        }

        const enabledButton = document.querySelector('.enabled-next');
        const disabledButton = document.querySelector('.disabled-next');
        const unsavedChanges = document.getElementById('unsaved-changes');

        if (hasValidQuestion) {
            enabledButton.classList.remove('hidden');
            disabledButton.classList.add('hidden');

            if (hasChanges) {
                unsavedChanges.classList.remove('hidden');
            }

            // Update form data to only include non-empty questions
            const form = document.createElement('form');
            let questionIndex = 0;
            inputs.forEach((input, index) => {
                if (input.value.trim()) {
                    const hiddenInput = document.createElement('input');
                    hiddenInput.type = 'hidden';
                    hiddenInput.name = `question_${questionIndex}`;
                    hiddenInput.value = input.value.trim();
                    form.appendChild(hiddenInput);
                    questionIndex++;
                }
            });

            // Update the htmx attributes on the enabled button
            enabledButton.setAttribute('hx-vals', JSON.stringify(Object.fromEntries(new FormData(form))));
        } else {
            enabledButton.classList.add('hidden');
            disabledButton.classList.remove('hidden');
            unsavedChanges.classList.add('hidden');
        }
    }

    updateSaveButtonState();
}

document.addEventListener('htmx:afterSwap', function(event) {
    if (event.detail.target.id === 'settings-content') {
        const questionsContainer = document.getElementById('questions-container');
        if (questionsContainer) {
            questionsContainer.addEventListener('input', function(e) {
                if (e.target.tagName === 'INPUT') {
                    updateSaveButtonState();
                }
            });
        }
    }
});

function updateRankerSaveButtonState() {
    const checkedInput = document.querySelector('input[name="ranker"]:checked');
    const unsavedChanges = document.getElementById('ranker-unsaved-changes');

    if (!checkedInput || !unsavedChanges) return;

    const originalValue = checkedInput.getAttribute('data-original') || '';
    const hasChanges = checkedInput.value !== originalValue;

    if (hasChanges) {
        unsavedChanges.classList.remove('hidden');
    } else {
        unsavedChanges.classList.add('hidden');
    }
}

function updateConnectionSaveButtonState() {
    const geminiToken = document.querySelector('input[name="gemini_token"]');
    const apiKeyFile = document.querySelector('input[name="api_key_file"]');

    const enabledButton = document.querySelector('#save-connection');
    const disabledButton = document.querySelector('#save-connection-disabled');
    const unsavedChanges = document.getElementById('connection-unsaved-changes');

    if (!geminiToken || !apiKeyFile) return;

    const isValid = geminiToken.value.trim() !== '' &&
                    (apiKeyFile.files.length > 0 || apiKeyFile.hasAttribute('data-has-file'));

    // Check if any field has changed from its original value
    const hasChanges = [geminiToken].some(input => {
        const originalValue = input.getAttribute('data-original') || '';
        return input.value.trim() !== originalValue.trim();
    }) || apiKeyFile.files.length > 0;

    if (isValid) {
        enabledButton.classList.remove('hidden');
        disabledButton.classList.add('hidden');

        if (hasChanges) {
            unsavedChanges.classList.remove('hidden');
        } else {
            unsavedChanges.classList.add('hidden');
        }
    } else {
        enabledButton.classList.add('hidden');
        disabledButton.classList.remove('hidden');
        unsavedChanges.classList.add('hidden');
    }
}

function updateApplicationPackageSaveButtonState() {
    const tenantName = document.querySelector('input[name="tenant_name"]');
    const appName = document.querySelector('input[name="app_name"]');
    const instanceName = document.querySelector('input[name="instance_name"]');
    const schema = document.querySelector('textarea[name="schema"]');

    const enabledButton = document.querySelector('#save-application-package');
    const disabledButton = document.querySelector('#save-application-package-disabled');
    const unsavedChanges = document.getElementById('application-package-unsaved-changes');

    if (!tenantName || !appName || !instanceName || !schema) return;

    const isValid = tenantName.value.trim() !== '' &&
                   appName.value.trim() !== '' &&
                   schema.value.trim() !== '';

    // Check if any field has changed from its original value
    const hasChanges = [tenantName, appName, instanceName, schema].some(input => {
        const originalValue = input.getAttribute('data-original') || '';
        return input.value.trim() !== originalValue.trim();
    });

    if (isValid) {
        enabledButton.classList.remove('hidden');
        disabledButton.classList.add('hidden');

        if (hasChanges) {
            unsavedChanges.classList.remove('hidden');
        } else {
            unsavedChanges.classList.add('hidden');
        }
    } else {
        enabledButton.classList.add('hidden');
        disabledButton.classList.remove('hidden');
        unsavedChanges.classList.add('hidden');
    }
}

function updatePromptSaveButtonState() {
    const promptTextarea = document.querySelector('textarea[name="prompt"]');
    const enabledButton = promptTextarea?.closest('form')?.querySelector('button[type="submit"]');
    const unsavedChanges = document.getElementById('prompt-unsaved-changes');

    if (!promptTextarea || !enabledButton || !unsavedChanges) return;

    // Check if the prompt has changed from its original value
    const originalValue = promptTextarea.getAttribute('data-original') || '';
    const hasChanges = promptTextarea.value.trim() !== originalValue.trim();
    const isEmpty = promptTextarea.value.trim() === '';

    // Disable button if empty, enable if not empty
    enabledButton.disabled = isEmpty;
    enabledButton.classList.toggle('opacity-50', isEmpty);
    enabledButton.classList.toggle('cursor-not-allowed', isEmpty);

    // Only show unsaved changes if there are changes and the prompt is not empty
    if (hasChanges && !isEmpty) {
        unsavedChanges.classList.remove('hidden');
    } else {
        unsavedChanges.classList.add('hidden');
    }
}

// Add event listener for HTMX after request
document.addEventListener('htmx:afterRequest', function(event) {
    // Check if it's the prompt form and the request was successful
    if (event.detail.elt.closest('form')?.matches('form[hx-post="/api/settings/prompt"]') &&
        event.detail.successful) {

        // Update the data-original attribute with the new value
        const promptTextarea = document.querySelector('textarea[name="prompt"]');
        if (promptTextarea) {
            promptTextarea.setAttribute('data-original', promptTextarea.value);
        }

        // Hide the unsaved changes message
        const unsavedChanges = document.getElementById('prompt-unsaved-changes');
        if (unsavedChanges) {
            unsavedChanges.classList.add('hidden');
        }
    }
});
