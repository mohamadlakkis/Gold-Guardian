function showTab(tabId) {
  // Hide all sections
  document.querySelectorAll('.tab-content').forEach(tab => tab.style.display = 'none');
  // Remove active class from all nav links
  document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
  // Show the selected tab
  document.getElementById(tabId).style.display = 'block';
  // Add active class to the selected nav link
  document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
}

window.onload = () => {
  // Retrieve the active tab from Flask, passed as a template variable
  const activeTab = "{{ active_tab }}";
  if (activeTab) {
      showTab(activeTab);  // Show the tab from Flask's active_tab variable
  } else {
      showTab('text-query-tab');  // Default to text-query-tab
  }
};
