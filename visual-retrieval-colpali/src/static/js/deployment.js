let isDeploying = false;

function handleVespaLogin(url) {
    window.open(url, '_blank');
    setTimeout(() => {
        const continueBtn = document.getElementById('continue-btn');
        if (continueBtn) {
            continueBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            continueBtn.disabled = false;

            // Remove any existing click listeners
            const newBtn = continueBtn.cloneNode(true);
            continueBtn.parentNode.replaceChild(newBtn, continueBtn);

            newBtn.addEventListener('click', () => {
                isDeploying = true;
                const modalContainer = document.getElementById('deployment-modal');
                htmx.ajax('GET', '/deployment-modal', {
                    target: '#deployment-modal',
                    swap: 'innerHTML'
                }).then(() => {
                    htmx.ajax('POST', '/api/deploy-part-2', {
                        target: '#deployment-modal',
                        swap: 'innerHTML'
                    });
                });
            });
        }
    }, 1000);
}

function closeDeploymentModal() {
    document.getElementById('deployment-modal').remove();
}

document.addEventListener('htmx:beforeRequest', function(event) {
    if (event.detail.requestConfig.path === '/api/deploy-part-1') {
        isDeploying = true;
        const modalContainer = document.createElement('div');
        modalContainer.id = 'deployment-modal';
        document.body.appendChild(modalContainer);

        htmx.ajax('GET', '/deployment-modal', {
            target: '#deployment-modal',
            swap: 'innerHTML'
        });
    }
});

document.addEventListener('htmx:afterRequest', function(event) {
    if (event.detail.requestConfig.path === '/api/deploy-part-1') {
        isDeploying = false;
        const modalContainer = document.getElementById('deployment-modal');

        if (modalContainer) {
            const response = JSON.parse(event.detail.xhr.response);

            if (response.status === 'success') {
                if (response.auth_url) {
                    htmx.ajax('GET', `/deployment-modal/login?auth_url=${encodeURIComponent(response.auth_url)}`, {
                        target: '#deployment-modal',
                        swap: 'innerHTML'
                    });
                } else {
                    htmx.ajax('GET', '/deployment-modal/success', {
                        target: '#deployment-modal',
                        swap: 'innerHTML'
                    });
                }
            } else {
                htmx.ajax('GET', '/deployment-modal/error', {
                    target: '#deployment-modal',
                    swap: 'innerHTML'
                });
            }
        }
    } else if (event.detail.requestConfig.path === '/api/deploy-part-2') {
        isDeploying = false;
        const modalContainer = document.getElementById('deployment-modal');

        if (modalContainer) {
            const response = JSON.parse(event.detail.xhr.response);

            if (response.status === 'success') {
                htmx.ajax('GET', '/deployment-modal/success', {
                    target: '#deployment-modal',
                    swap: 'innerHTML'
                });
            } else {
                htmx.ajax('GET', '/deployment-modal/error', {
                    target: '#deployment-modal',
                    swap: 'innerHTML'
                });
            }
        }
    }
});

window.addEventListener('beforeunload', function(e) {
    if (isDeploying) {
        e.preventDefault();
        e.returnValue = 'Application deployment is in progress. Are you sure you want to leave?';
        return e.returnValue;
    }
});
