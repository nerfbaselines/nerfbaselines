function init() {
  // Init is called with loaded document
  //
  // Table
  // - Expand rows
  const expandToggles = document.querySelectorAll(".table .table__expandtoggle[data-expand-id]");
  for (var i = 0; i < expandToggles.length; i++) {
    const expand = expandToggles[i];
    expand.addEventListener("click", function(e) {
      const obj = e.target.closest("[data-expand-id]");
      obj.classList.toggle("expanded");
      if (obj.classList.contains("expanded"))
        obj.attributes["title"].value = obj.attributes["title"].value.replace("Expand", "Collapse");
      else
        obj.attributes["title"].value = obj.attributes["title"].value.replace("Collapse", "Expand");
      var expanded = obj.classList.contains("expanded");
      const expandId = obj.getAttribute("data-expand-id");
      const rows = obj.closest(".table").querySelectorAll("tr[data-expand-by='" + expandId + "']")
      for (var j = 0; j < rows.length; j++) {
        const row = rows[j];
        // Remove/add style="display: none" from the row
        if (expanded) {
          row.style.removeProperty("display");
        } else {
          row.style.display = "none";
        }
      }
    });
  }

  // Table
  // - Sort
  const sortButtons = document.querySelectorAll(".table .table__sortbutton");
  for (var i = 0; i < sortButtons.length; i++) { 
    const sortButton = sortButtons[i];
    sortButton.addEventListener("click", function(e) {
      // If was sort-active, toggle direction
      // If was not sort-active, set to active and keep direction
      if (sortButton.classList.contains("sort-active")) {
        // Toggle direction
        sortButton.classList.toggle("sort-desc");
        if (sortButton.classList.contains("sort-desc"))
          sortButton.attributes["title"].value = sortButton.attributes["title"].value.replace("descending", "ascending");
        else
          sortButton.attributes["title"].value = sortButton.attributes["title"].value.replace("ascending", "descending");
      } else {
        // Set to active
        sortButton.closest("tr").querySelectorAll(".table__sortbutton").forEach(function(button) {
          button.classList.remove("sort-active");
          // Switch direction in title
          if (button.classList.contains("sort-desc"))
            button.attributes["title"].value = button.attributes["title"].value.replace("ascending", "descending");
          else
            button.attributes["title"].value = button.attributes["title"].value.replace("descending", "ascending");
        });
        sortButton.classList.add("sort-active");
        if (sortButton.classList.contains("sort-desc"))
          sortButton.attributes["title"].value = sortButton.attributes["title"].value.replace("descending", "ascending");
        else
          sortButton.attributes["title"].value = sortButton.attributes["title"].value.replace("ascending", "descending");
      }
      const dir = sortButton.classList.contains("sort-desc") ? "desc" : "asc";
      // Read the data first (index of e.target.closest("th"))
      const columnId = e.target.closest("th").cellIndex;
      const tbody = e.target.closest("table").querySelector("tbody");
      const rowData = Array.from(tbody.children);
      if (rowData.length < 2) {
        return;
      }
      // Sort the data while ignoring rows with [data-expand-by]
      const dataRanges = [];
      for (var j = 0; j < rowData.length; j++) {
        const row = rowData[j];
        if (row.hasAttribute("data-expand-by")) {
          dataRanges[dataRanges.length-1].length += 1;
        } else {
          dataRanges.push({
            start: j,
            length: 1,
            value: parseInt(row.children[columnId].attributes["data-sort-value"].value)
          });
        }
      }
      dataRanges.sort(function(a, b) {
        const value = Math.sign(a.value - b.value);
        return dir === "asc" ? value : -value;
      });
      // Reorder the rows
      const newRows = [];
      for (var j = 0; j < dataRanges.length; j++) {
        const range = dataRanges[j];
        for (var k = 0; k < range.length; k++) {
          newRows.push(rowData[range.start + k]);
        }
      }

      // Reorder the HTML elements
      tbody.insertBefore(newRows[newRows.length-1], null);
      for (var j = newRows.length-2; j>=0; --j) {
        tbody.insertBefore(newRows[j], newRows[j+1]);
      }
    });
  }
  // Table
  // - Copy
  document.querySelectorAll(".table .table__allowcopy").forEach(function (copycell) {
    copycell.addEventListener("click", function(e) {
      const text = copycell.innerText;
      navigator.clipboard.writeText(text);
    });
  });
  // - Tooltips
  function updateInfobox(tooltip) {
    const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0)
    const td = tooltip.closest("td");
    const tooltipbb = tooltip.getBoundingClientRect();
    const bodyLeft = document.body.getBoundingClientRect().x;
    if (tooltipbb.width >= vw*0.70) {
      tooltip.style.left = (vw*0.1-bodyLeft) + "px";
      return;
    }
    const tdbb = td.querySelector("span").getBoundingClientRect();
    let clientLeft = tdbb.x;
    clientLeft = Math.min(Math.max(clientLeft, 0), vw-tooltipbb.width);
    tooltip.style.left = (clientLeft-bodyLeft) + "px";
  }
  let currentInfobox = null;
  const tooltips = document.querySelectorAll(".table .table__infobox");
  const tableContainers = document.querySelectorAll(".table-container");
  const infoboxes = document.querySelectorAll(".table .table__infobox");
  for (var i=0; i<tableContainers.length; ++i) {
    tableContainers[i].addEventListener("scroll", function(e) {
      e.target.querySelectorAll(".table__infobox").forEach(function(tooltip) {
        updateInfobox(tooltip);
      });
    });
  }
  window.addEventListener("resize", function(e) {
    if (currentInfobox) {
      updateInfobox(currentInfobox);
    }
  });
  infoboxes.forEach(function(tooltip) {
    currentInfobox = tooltip;
    tooltip.closest("td").addEventListener("mouseenter", function(e) { updateInfobox(tooltip); });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

