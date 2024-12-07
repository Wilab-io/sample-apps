let isDeploying = false;

document.addEventListener('htmx:beforeRequest', function(event) {
    if (event.detail.requestConfig.path === '/api/deploy') {
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
    if (event.detail.requestConfig.path === '/api/deploy') {
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
