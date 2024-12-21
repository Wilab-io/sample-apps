let isProcessing = false;

document.addEventListener('htmx:beforeRequest', function (event) {
  if (event.detail.requestConfig.path === '/upload-files') {
    isProcessing = true;
    const modalContainer = document.createElement('div');
    modalContainer.id = 'document-processing-modal';
    document.body.appendChild(modalContainer);

    htmx.ajax('GET', '/document-processing-modal', {
      target: '#document-processing-modal',
      swap: 'innerHTML'
    });
  }
});

document.addEventListener('htmx:afterRequest', function (event) {
  if (event.detail.requestConfig.path === '/upload-files') {
    isProcessing = false;
    const modalContainer = document.getElementById('document-processing-modal');

    if (modalContainer) {
      try {
        const response = JSON.parse(event.detail.xhr.response);

        if (response.status === 'success') {
          modalContainer.remove();
          window.location.href = '/my-documents';
        } else {
          const errorMessage = response.message || 'An unknown error occurred';
          htmx.ajax('GET', '/document-processing-modal/error?message=' + encodeURIComponent(errorMessage), {
            target: '#document-processing-modal',
            swap: 'innerHTML'
          });
        }
      } catch (e) {
        console.error('Error parsing response:', e);
        htmx.ajax('GET', '/document-processing-modal/error?message=' + encodeURIComponent('Failed to process server response'), {
          target: '#document-processing-modal',
          swap: 'innerHTML'
        });
      }
    }
  }
});

document.addEventListener('htmx:beforeRequest', function (event) {
  if (event.detail.requestConfig.path.startsWith('/delete-document/')) {
    isProcessing = true;
    const modalContainer = document.createElement('div');
    modalContainer.id = 'document-deleting-modal';
    document.body.appendChild(modalContainer);

    htmx.ajax('GET', '/document-deleting-modal', {
      target: '#document-deleting-modal',
      swap: 'innerHTML'
    });
  }
});

document.addEventListener('htmx:afterRequest', function (event) {
  if (event.detail.requestConfig.path.startsWith('/delete-document/')) {
    isProcessing = false;
    const modalContainer = document.getElementById('document-deleting-modal');

    if (modalContainer) {
      try {
        const response = JSON.parse(event.detail.xhr.response);

        if (response.status === 'success') {
          modalContainer.remove();
          window.location.href = '/my-documents';
        } else {
          const errorMessage = response.message || 'An unknown error occurred';
          htmx.ajax('GET', '/document-deleting-modal/error?message=' + encodeURIComponent(errorMessage), {
            target: '#document-deleting-modal',
            swap: 'innerHTML'
          });
        }
      } catch (e) {
        console.error('Error parsing response:', e);
        htmx.ajax('GET', '/document-deleting-modal/error?message=' + encodeURIComponent('Failed to process server response'), {
          target: '#document-deleting-modal',
          swap: 'innerHTML'
        });
      }
    }
  }
});

window.addEventListener('beforeunload', function (e) {
  if (isProcessing) {
    e.preventDefault();
    e.returnValue = 'Document processing is in progress. Are you sure you want to leave?';
    return e.returnValue;
  }
});
